# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed
import torch.nn.functional as F

from torch.nn.init import trunc_normal_

from models.sam2.modeling.sam.mask_decoder import MaskDecoder
from models.sam2.modeling.sam.prompt_encoder import PromptEncoder
from models.sam2.modeling.sam.transformer import TwoWayTransformer
from models.sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames
from models.model_utils import DecoderOutput

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class EfficientBase(torch.nn.Module):
    def __init__(
        self,
        image_encoder,
        audio_encoder,
        text_encoder,
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder


