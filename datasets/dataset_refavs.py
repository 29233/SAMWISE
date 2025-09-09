import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pdb
from pathlib import Path
import sys
import os
import random
from pathlib import Path
from torchvision import transforms
from collections import defaultdict
import cv2
from transformers import AutoImageProcessor, AutoTokenizer, AutoModel
from PIL import Image

from towhee import pipe, ops
from transformers import pipeline

# logger = log_agent('audio_recs.log')

import pickle as pkl


class REFAVS(Dataset):
    def __init__(self, split='train', args=None):
        # metadata: train/test/val
        self.refavs_path = args.refavs_path
        meta_path = f'{self.refavs_path}/metadata.csv'
        metadata = pd.read_csv(meta_path, header=0)
        self.split = split
        self.metadata = metadata[metadata['split'] == split]  # split= train,test,val.

        self.media_path = f'{self.refavs_path}/media'
        self.label_path = f'{self.refavs_path}/gt_mask'
        self.frame_num = args.num_frames
        self.text_max_len = args.text_max_len

        # modalities processor/pipelines
        self.img_process = AutoImageProcessor.from_pretrained(args.m2f_model)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        df_one_video = self.metadata.iloc[idx]
        vid, uid, fid, exp = df_one_video['vid'], df_one_video['uid'], df_one_video['fid'], df_one_video[
            'exp']  # uid for vid.
        vid = uid.rsplit('_', 2)[0]  # TODO: use encoded id.

        img_recs = []
        mask_recs = []
        images = []

        rec_audio = f'{self.media_path}/{vid}/audio.wav'
        rec_text = exp

        for _idx in range(self.frame_num):  # set frame_num as the batch_size
            # frame 
            path_frame = f'{self.media_path}/{vid}/frames/{_idx}.jpg'  # image
            image = Image.open(path_frame)
            image_sizes = image.size[::-1]
            image_inputs = self.img_process(image, return_tensors="pt")  # singe frame rec
            image_inputs = image_inputs.data['pixel_values']
            # mask label
            path_mask = f'{self.label_path}/{vid}/fid_{fid}/0000{_idx}.png'  # new
            mask_cv2 = cv2.imread(path_mask)
            mask_cv2 = cv2.resize(mask_cv2, (256, 256))
            mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)
            gt_binary_mask = torch.as_tensor(mask_cv2 > 0, dtype=torch.float32)

            # video frames collect
            img_recs.append(image_inputs)
            mask_recs.append(gt_binary_mask)

        img_recs = torch.cat(img_recs)
        mask_recs = torch.stack(mask_recs)
        target = {
            'frames_idx': list(range(0,10)),  # [T,]
            'masks': mask_recs,  # [T, H, W]
            'caption': exp,
            'orig_size': torch.as_tensor(image_sizes),
            'size': torch.as_tensor(image_sizes),
            'video_id': vid,
            'exp_id': idx,
            'mask_id': fid
        }
        return img_recs, target

def build(image_set, args):
    root = Path(args.refavs_path)

    assert root.exists(), f'provided mevis path {root} does not exist'
    dataset = REFAVS(image_set, args)
    return dataset