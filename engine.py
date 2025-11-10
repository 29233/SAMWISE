"""
Train and eval functions used in main.py
Modified from DETR (https://github.com/facebookresearch/detr)
"""
import numpy as np
import math
import os
import sys
from typing import Iterable
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tools.metrics import calculate_precision_at_k_and_iou_metrics
from tqdm import tqdm
import util.misc as utils
from torch.nn import functional as F
from models.segmentation import loss_masks
from tools.metrics import mask_iou, Eval_Fmeasure, metric_s_for_null
from util.misc import interpolate
from PIL import Image


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    lr_scheduler=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    step=0

    for samples, captions, audios, targets in metric_logger.log_every(data_loader, print_freq, header):
        step+=1
        model.train()
        samples = samples.to(device)
        # captions = [t["caption"] for t in targets]
        outputs = model(samples, captions, audios, targets)
        losses = {}
        # seg_loss = loss_masks(torch.cat(outputs["masks"]), targets, num_frames=samples.tensors.shape[1])
        seg_loss = loss_masks(outputs[0], targets, num_frames=samples.tensors.shape[1])
        losses.update(seg_loss)
        if args.use_cme_head and "pred_cme_logits" in outputs:
            weight = torch.tensor([1., 2.]).to(device)
            CME_loss = F.cross_entropy(torch.cat(outputs["pred_cme_logits"]), ignore_index=-1,
                                        target=torch.tensor(outputs["cme_label"]).long().to(device),
                                        weight=weight)
            losses.update({"CME_loss": CME_loss if not CME_loss.isnan() else torch.tensor(0).to(device)})

        loss_dict = losses
        losses = sum(loss_dict[k] for k in loss_dict.keys())
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v for k, v in loss_dict_reduced.items()}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)

        optimizer.step()
        lr_scheduler.step()


        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, data_loader, device, args):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 50
    if data_loader.dataset.split != 'test_n':
        for samples, captions, audios, targets in metric_logger.log_every(data_loader, print_freq, header):
            with torch.no_grad():
                samples = samples.to(device)
                targets = utils.targets_to(targets, device)
                outputs = model(samples, captions, audios, targets)
                pred_masks = outputs[0].squeeze()
                gt_masks = targets[0]['masks']
                iou = mask_iou(pred_masks, gt_masks)
                f_score = Eval_Fmeasure(pred_masks, gt_masks)
                metric_logger.update(iou=iou)
                metric_logger.update(F_score=f_score)
                if args.save_pred_masks:
                    temp_pred = torch.sigmoid(outputs[0])
                    save_mask = interpolate(temp_pred, size=tuple(targets[0]['size']), mode="bilinear",
                                            align_corners=False)
                    save_mask = (save_mask > 0.4).int()
                    # TODO: only support batch size = 1
                    for target in  targets:
                        mask_path = f"{args.output_dir}/{args.split}/{target['video_id']}/fid_{target['mask_id']}/{target['class_id']}_{target['sample_id']}"
                        if not os.path.exists(mask_path):
                            os.makedirs(mask_path)
                        for idx, mask in enumerate(save_mask):
                            if isinstance(mask, torch.Tensor):
                                if mask.is_cuda:
                                    mask = mask.cpu()
                                mask = mask.numpy()
                            mask_img = Image.fromarray((np.squeeze(mask) * 255).astype(np.uint8))
                            mask_img.save(os.path.join(mask_path, f'0000{idx}.png'))
                        with open(f"{mask_path}/meta.txt", "w") as f:
                            f.write(f"caption: {target['caption']}\n")
            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
    else:
        for samples, captions, audios, targets in metric_logger.log_every(data_loader, print_freq, header):
            with torch.no_grad():
                samples = samples.to(device)
                targets = utils.targets_to(targets, device)
                outputs = model(samples, captions, audios, targets)
                pred_masks = outputs[0].squeeze()
                S_score = metric_s_for_null(pred_masks)
                metric_logger.update(S_score=S_score)
                if args.save_pred_masks:
                    temp_pred = torch.sigmoid(outputs[0])
                    save_mask = interpolate(temp_pred, size=tuple(targets[0]['size']), mode="bilinear",
                                            align_corners=False)
                    save_mask = (save_mask > 0.4).int()
                    # TODO: only support batch size = 1
                    for target in  targets:
                        mask_path = f"{args.output_dir}/{args.split}/{target['video_id']}/fid_{target['mask_id']}/{target['class_id']}_{target['sample_id']}"
                        if not os.path.exists(mask_path):
                            os.makedirs(mask_path)
                        for idx, mask in enumerate(save_mask):
                            if isinstance(mask, torch.Tensor):
                                if mask.is_cuda:
                                    mask = mask.cpu()
                                mask = mask.numpy()
                            mask_img = Image.fromarray((np.squeeze(mask) * 255).astype(np.uint8))
                            mask_img.save(os.path.join(mask_path, f'0000{idx}.png'))
                        with open(f"{mask_path}/meta.txt", "w") as f:
                            f.write(f"caption: {target['caption']}\n")
            # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)

@torch.no_grad()
def evaluate_org(model, postprocessors, data_loader, device, args):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    predictions = []
    ious = []
    F_scores=[]
    with tqdm(data_loader) as pbar:
        for samples, captions, audios, targets in pbar:
            # dataset_name = targets[0]["dataset_name"]
            dataset_name = 'refavs'
            samples = samples.to(device)
            # captions = [t["caption"] for t in targets]
            targets = utils.targets_to(targets, device)

            outputs = model(samples, captions, audios, targets)
            # pred_masks = outputs['pred_masks']
            pred_masks = outputs[0].squeeze()
            gt_masks = targets[0]['masks']
            iou = mask_iou(pred_masks, gt_masks)
            f_score = Eval_Fmeasure(pred_masks, gt_masks)
            ious.append(iou)
            F_scores.append(f_score)
            miou = sum(ious) / len(ious)
            mF = sum(F_scores) / len(F_scores)
            pbar.set_postfix(miou=f"{miou:.4f}", mF=f"{mF:.4f}")
            pbar.update()
    miou = sum(ious) / len(ious)
    mF = sum(F_scores) / len(F_scores)
    print("miou:", miou)
    print("mF:", mF)