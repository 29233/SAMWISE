# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import torch
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder



class MaskFormerHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    def __init__(
        self,
        idx_features: List[str],
        channels: List[int],
        strides: List[int],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()

        input_shape={
            k: ShapeSpec(channels=ch, stride=s)
            for k, ch, s in zip(idx_features, channels, strides)
        }
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]


        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

    # @classmethod
    # def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
    #     # figure out in_channels to transformer predictor
    #     if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
    #         transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
    #     elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
    #         transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
    #     elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
    #         transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
    #     else:
    #         transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels
    #
    #     return {
    #         "input_shape": {
    #             k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
    #         },
    #         "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
    #         "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
    #         "pixel_decoder": build_pixel_decoder(cfg, input_shape),
    #         "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
    #         "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
    #         "transformer_predictor": build_transformer_decoder(
    #             cfg,
    #             transformer_predictor_in_channels,
    #             mask_classification=True,
    #         ),
    #     }

    def forward(self, features, mask=None):
        return self.layers(features, mask)

    def layers(self, features, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask)
        else:
            if self.transformer_in_feature == "transformer_encoder":
                assert (
                    transformer_encoder_features is not None
                ), "Please use the TransformerEncoderPixelDecoder."
                predictions = self.predictor(transformer_encoder_features, mask_features, mask)
            elif self.transformer_in_feature == "pixel_embedding":
                predictions = self.predictor(mask_features, mask_features, mask)
            else:
                predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        return predictions



# generate a script to test MaskFormerHead
if __name__ == "__main__":
    head = MaskFormerHead(
        input_shape={
            "p2": ShapeSpec(channels=256, stride=4),
            "p3": ShapeSpec(channels=512, stride=8),
            "p4": ShapeSpec(channels=1024, stride=16),
            "p5": ShapeSpec(channels=2048, stride=32),
        },
        num_classes=80,
        pixel_decoder=nn.Identity(),
        transformer_predictor=nn.Identity(),
        transformer_in_feature="p2",
    )
    features = {
        "p2": torch.randn(2, 256, 128, 128),
        "p3": torch.randn(2, 512, 64, 64),
        "p4": torch.randn(2, 1024, 32, 32),
        "p5": torch.randn(2, 2048, 16, 16),
    }
    outputs = head(features)
    print(outputs)  # should be a tuple of (class_logits, mask_logits)
