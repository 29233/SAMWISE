'''
Inference code for SAMWISE, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
import os
from PIL import Image
import torch.nn.functional as F
import json
from tqdm import tqdm
import sys
from pycocotools import mask as cocomask
from tools.colormap import colormap
import opts
from models.samravs import build_samravs
from util.misc import on_load_checkpoint
from datasets import build_dataset
from tools.metrics import db_eval_boundary, db_eval_iou
from datasets.transform_utils import VideoEvalDataset
from torch.utils.data import DataLoader
from os.path import join
from datasets.transform_utils import vis_add_mask
from towhee import pipe


# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()


def main(args):
    args.batch_size = 1
    print("Inference only supports for batch size = 1") 
    print(args)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    utils.init_distributed_mode(args)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, 'Annotations')
    os.makedirs(save_path_prefix, exist_ok=True)
    args.log_file = join(output_dir, 'log.txt')
    with open(args.log_file, 'w') as fp:
        fp.writelines(" ".join(sys.argv)+'\n')
        fp.writelines(str(args.__dict__)+'\n\n')        

    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    if args.visualize:
        os.makedirs(save_visualize_path_prefix, exist_ok=True)

    start_time = time.time()
    # model
    model = build_samravs(args)
    device = torch.device(args.device)
    model.to(device)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('number of params:', n_parameters)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if list(checkpoint['model'].keys())[0].startswith('module'):
            checkpoint['model'] = {k.replace('module.', ''): v for k, v in checkpoint['model'].items()}        
        checkpoint = on_load_checkpoint(model_without_ddp, checkpoint)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    dataset_train = build_dataset(args.dataset_file, image_set=args.split, args=args)
    testloader = DataLoader(dataset_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    print('Start inference')
    result = save_with_stamp(model, testloader, args)

    if args.split == 'valid_u':
        J_score, F_score, JF = result[0], result[1], result[2]
        out_str = f'J&F: {JF}\tJ: {J_score}\tF: {F_score}'
        with open(args.log_file, 'a') as fp:
            fp.writelines(out_str+'\n')

    end_time = time.time()
    total_time = end_time - start_time

    print("Total inference time: %.4f s" %(total_time))


def save_with_stamp(model, test_loader, args):
    model.eval()

    null_s_list = []
    progress_bar = tqdm(test_loader, total=len(test_loader))

    total_data_time = 0.0
    total_model_time = 0.0
    total_metric_time = 0.0
    total_save_time = 0.0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            # --- 1. 数据加载阶段 ---
            start_data = time.time()
            sample, target = batch_data
            mask_recs, fids, uids = batch_data
            end_data = time.time()
            total_data_time += end_data - start_data

            # --- 2. 模型推理阶段 ---
            start_model = time.time()
            _, vid_preds = model(sample)
            mask_recs = [torch.stack(mask_rec, dim=0) for mask_rec in mask_recs]
            vid_preds_t = torch.stack(vid_preds, dim=0).squeeze().cuda().view(-1, 1, 256, 256)
            vid_masks_t = torch.stack(mask_recs, dim=0).squeeze().cuda().view(-1, 1, 256, 256)
            pre_mask = vid_preds_t.view(len(uids), -1, 256, 256)
            pred_mask = torch.sigmoid(pre_mask).cpu().numpy()
            pred_mask = (pred_mask > 0.4).astype(np.uint8)
            end_model = time.time()
            total_model_time += end_model - start_model

            # --- 3. 指标计算阶段 ---
            start_metric = time.time()
            miou = mask_iou(vid_preds_t, vid_masks_t)
            F_score = Eval_Fmeasure(vid_preds_t, vid_masks_t, './logger', device=f'cuda:{args.gpu_id}')
            avg_meter_miou.add({'miou': miou})
            avg_meter_F.add({'F_score': F_score})
            end_metric = time.time()
            total_metric_time += end_metric - start_metric

            # --- 4. mask 保存阶段 ---
            start_save = time.time()
            for idx, (sample, uid, fid) in enumerate(zip(pred_mask, uids, fids)):
                mask_path = f"{args.save_path}/{args.task}/{args.val}/{uid}/fid_{fid}"
                if not os.path.exists(mask_path):
                    os.makedirs(mask_path)
                for id, mask in enumerate(sample):
                    mask_img = Image.fromarray(mask * 255)
                    mask_img.save(os.path.join(mask_path, f'0000{id}.png'))
            end_save = time.time()
            total_save_time += end_save - start_save

            # --- 打印总耗时 ---
            print(f"  Data Loading Time: {total_data_time:.4f}s")
            print(f"  Model Inference Time: {total_model_time:.4f}s")
            print(f"  Metric Calculation Time: {total_metric_time:.4f}s")
            print(f"  Mask Saving Time: {total_save_time:.4f}s")
            print(f"  Total Time: {total_data_time + total_model_time + total_metric_time + total_save_time:.4f}s")

        miou_epoch = (avg_meter_miou.pop('miou')).item()
        F_epoch = (avg_meter_F.pop('F_score'))

    return miou_epoch, F_epoch


def get_current_metrics(out_dict):
    j = [out_dict[x][0] for x in out_dict]
    f = [out_dict[x][1] for x in out_dict]

    J_score = np.mean(j)
    F_score = np.mean(f)
    JF = (np.mean(j) + np.mean(f)) / 2
    return J_score, F_score, JF

def mask_iou(pred, target, eps=1e-7, size_average=True):
    r"""
        param:
            pred: size [N x H x W]
            target: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """
    # return mask_iou_224(pred, target, eps=1e-7)
    NF, bsz, H, W = pred.shape
    pred = pred.view(NF*bsz, H, W)
    target = target.view(NF*bsz, H, W)
    assert len(pred.shape) == 3 and pred.shape == target.shape

    N = pred.size(0)
    num_pixels = pred.size(-1) * pred.size(-2)
    no_obj_flag = (target.sum(2).sum(1) == 0)

    temp_pred = torch.sigmoid(pred)
    pred = (temp_pred > 0.4).int()
    inter = (pred * target).sum(2).sum(1)
    union = torch.max(pred, target).sum(2).sum(1)

    inter_no_obj = ((1 - target) * (1 - pred)).sum(2).sum(1)
    inter[no_obj_flag] = inter_no_obj[no_obj_flag]
    union[no_obj_flag] = num_pixels

    iou = torch.sum(inter / (union + eps)) / N

    return iou

def _eval_pr(y_pred, y, num, device='cuda'):
    if device.startswith('cuda'):
        prec, recall = torch.zeros(num).to(y_pred.device), torch.zeros(num).to(y_pred.device)
        thlist = torch.linspace(0, 1 - 1e-10, num).to(y_pred.device)
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)

    return prec, recall

def Eval_Fmeasure(pred, gt, measure_path, pr_num=255, device='cuda'):
    r"""
        param:
            pred: size [N x H x W]
            gt: size [N x H x W]
        output:
            iou: size [1] (size_average=True) or [N] (size_average=False)
    """

    pred = torch.sigmoid(pred)
    N = pred.size(0)
    beta2 = 0.3
    avg_f, img_num = 0.0, 0
    score = torch.zeros(pr_num)


    for img_id in range(N):
        if torch.mean(gt[img_id]) == 0.0:
            continue
        prec, recall = _eval_pr(pred[img_id], gt[img_id], pr_num, device=device)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
        avg_f += f_score
        img_num += 1
        score = avg_f / img_num

    return score.max().item()

class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1

    def get(self, *keys):
        if len(keys) == 1:
            return self.__data[keys[0]][0] / self.__data[keys[0]][1]
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    avg_meter_miou = AverageMeter('miou')
    avg_meter_F = AverageMeter('F_score')
    parser = argparse.ArgumentParser('SAMWISE evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    name_exp = args.name_exp
    args.output_dir = os.path.join(args.output_dir, name_exp)

    main(args)
