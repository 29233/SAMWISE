import torch
from torch import nn
from util.misc import nested_tensor_from_videos_list, NestedTensor
from models.CMT_adapter import CMT_adapter
from hydra import compose, initialize
import spacy
from models.sam2.modeling.sam2_utils import preprocess
from hydra.utils import instantiate
from omegaconf import OmegaConf
import os
import py3_wget
from models.conditional_memory_encoder import ConditionalMemoryEncoder
from fairseq.models.roberta import RobertaModel
from models.model_utils import BackboneOutput, DecoderOutput, get_same_object_labels
from transformers import RobertaTokenizerFast


class SAMRAVS(nn.Module):
    def __init__(self,
                 image_encoder_embed_dim,
                 text_encoder,
                 text_encoder_embed_dim,
                 audio_encoder,
                 audio_encoder_embed_dim,
                 fusion_stages_txt,
                 fusion_stages,
                 image_size,
                 sam,
                 conditional_memory_encoder,
                 adapter_dim,
                 args
                 ):
        super().__init__()

        self.text_encoder = text_encoder
        self.tokenizer = RobertaTokenizerFast.from_pretrained('pretrain/roberta')
        self.text_encoder_embed_dim = text_encoder_embed_dim

        # self.conditional_memory_encoder = conditional_memory_encoder

        self.audio_encoder = audio_encoder
        #
        self.motion_prompt = args.motion_prompt
        if args.motion_prompt:
            self.nlp_dict = spacy.load("en_core_web_sm")

        self.memory_bank = {}

        self.image_size = image_size

    @staticmethod
    def preprocess_visual_features(samples, image_size):
        # zero padding
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_videos_list(samples)
        samples, masks = samples.decompose()
        B, T, C, H, W = samples.shape
        samples = samples.view(B * T, C, H, W)
        orig_size = [tuple(x.shape[-2:]) for x in samples]
        samples = torch.stack([preprocess(x, image_size) for x in samples], dim=0)
        BT = (B, T)
        return samples, BT, orig_size

    def preprocess_text_features(self, captions):
        batch_encoding_text = self.tokenizer(captions, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(batch_encoding_text['input_ids']).cuda()
        attention_mask = torch.tensor(batch_encoding_text['attention_mask']).eq(0).cuda()
        text_encoder = self.text_encoder.model.encoder.sentence_encoder
        has_pads = (torch.tensor(input_ids.device.type == "xla") or attention_mask.any())
        x, encoder_embedding = text_encoder.forward_embedding(input_ids, None)
        txt = x * (1 - attention_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x))
        # txt = x.transpose(0, 1)  # B x T x C -> T x B x C
        return txt, attention_mask, input_ids

    def preprocess_audio_features(self, wav_path):
        """ wav string path. """
        emb = torch.tensor(self.audio_encoder(wav_path).get()[0])
        if emb.shape[0] == 10:
            emb = emb.repeat(2, 1)
        # print(len(emb))
        return emb

    def _forward_fpn(self, vis_outs):
        features, pos = self.sam.image_encoder.neck(vis_outs)

        # Discard the lowest resolution features
        features, pos = features[: -1], pos[: -1]
        image_embedding = features[-1]

        backbone_out = {
            "vision_features": image_embedding,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        backbone_out["backbone_fpn"][0] = self.sam.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.sam.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        return backbone_out

    def forward(self, samples, captions, audios, targets):
        samples, (B, T), orig_size = self.preprocess_visual_features(samples, self.image_size)
        backbone_out = self.sam.forward_images(samples)
        txt, attention_mask, input_ids = self.preprocess_text_features(captions)
        txt_state = txt[:,0]
        audio_embs = [self.preprocess_audio_features(wav_path) for wav_path in audios]
        audio_embs = torch.stack(audio_embs, dim=0).cuda()
        _, vision_feats, vision_pos_embeds, feat_sizes = self.sam._prepare_backbone_features(backbone_out)
        backbone_output = BackboneOutput(
            B=B,
            T=T,
            vision_feats=vision_feats,
            vision_pos_embeds=vision_pos_embeds,
            feat_sizes=feat_sizes,
            orig_size=orig_size,
            state=txt_state,
            audio_feats=audio_embs
        )
        outputs = {"masks": []}
        for video_record in range(B):
            if self.training or T==1: # T == 1 for pre-training, no propagation from memory bank
                self.memory_bank, self.last_frame_cme_applied = {}, 0
            elif targets[0]['frame_ids'][0] == 0:  # it's the first frame of a new video
                self.memory_bank, self.last_frame_cme_applied = {}, 0

            for frame_idx in range(T):
                idx = video_record * T + frame_idx
                # use relative IDX in the clip
                if self.training or T==1:  # T == 1 for pre-training, no propagation from memory bank
                    memory_idx = frame_idx
                # use absolute IDX in the video
                else:
                    memory_idx = targets[0]['frame_ids'][frame_idx]

                current_vision_feats = backbone_output.get_current_feats(idx)
                decoder_out_w_mem: DecoderOutput = self.compute_decoder_out_w_mem(backbone_output, idx, memory_idx,
                                                                                  self.memory_bank)
            mem_dict_w_mem = self.compute_memory_bank_dict(decoder_out_w_mem, current_vision_feats,
                                                           backbone_output.feat_sizes)
            self.memory_bank[memory_idx] = mem_dict_w_mem
            outputs["masks"].append(decoder_out_w_mem.masks)

        masks = torch.cat(outputs["masks"])
        if self.training:
            return outputs
        else:
            return {"pred_masks": masks.squeeze(1)}




