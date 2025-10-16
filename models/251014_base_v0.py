import torch
from torch import nn
from util.misc import nested_tensor_from_videos_list, NestedTensor
from models.sam2.modeling.sam2_utils import preprocess


class EfficientBase(nn.Module):
    def __init__(self,
                 image_encoder,
                 text_encoder,
                 text_tokenizer,
                 audio_encoder,
                 interactor,
                 decoder,
                 image_size,
                 audio_feature_dim,
                 text_feature_dim,
                 image_encoder_checkpoint
                 ):
        super().__init__()
        self. image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.audio_encoder = audio_encoder
        self.interactor = interactor
        self.decoder = decoder
        self.image_size = image_size

        self.audio_projecter = nn.Linear(audio_feature_dim, self.interactor.d_model)
        self.text_projecter = nn.Linear(text_feature_dim, self.interactor.d_model)

        # load image encoder parameters
        self.image_encoder.load_state_dict(torch.load(image_encoder_checkpoint, map_location='cpu'), strict=True)
        # set encoders frozen
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.audio_encoder.parameters():
            param.requires_grad = False

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
        batch_encoding_text = self.text_tokenizer(captions, add_special_tokens=True, padding=True)   # 0:BOS 1:padding 2:EOS
        input_ids = torch.tensor(batch_encoding_text['input_ids']).cuda()
        attention_mask = torch.tensor(batch_encoding_text['attention_mask']).eq(0).cuda()
        text_encoder = self.text_encoder.model.encoder.sentence_encoder
        has_pads = (torch.tensor(input_ids.device.type == "xla") or attention_mask.any())
        x, encoder_embedding = text_encoder.forward_embedding(input_ids, None)
        txt = x * (1 - attention_mask.unsqueeze(-1).type_as(x) * has_pads.type_as(x))
        # txt = x.transpose(0, 1)  # B x T x C -> T x B x C
        return txt, attention_mask, input_ids

    def preprocess_audio_features(self, wav_path):
        audio = vggish_input.wavfile_to_examples(wav_path)
        if audio.shape[0] != 10:
            print('lm.shape: ', audio.shape)
            N_SECONDS, CHANNEL, N_BINS, N_BANDS = audio.shape
            new_lm_tensor = torch.zeros(5, CHANNEL, N_BINS, N_BANDS)
            new_lm_tensor[:N_SECONDS] = audio
            new_lm_tensor[N_SECONDS:] = audio[-1].repeat(5 - N_SECONDS, 1, 1, 1)
            audio = new_lm_tensor
        emb = self.audio_encoder(audio.to(self.audio_encoder.device))
        # print(len(emb))
        return emb

    def forward(self, samples, captions, audios, targets):
        samples, (B, T), orig_size = self.preprocess_visual_features(samples, self.image_size)
        backbone_out = self.image_encoder(samples)
        txt, attention_mask, input_ids = self.preprocess_text_features(captions)
        txt_state = txt[:,0]
        audio_embs = [self.preprocess_audio_features(wav_path) for wav_path in audios]
        audio_embs = torch.stack(audio_embs, dim=0).unsqueeze(2).cuda()
        audio_embs = self.audio_projecter(audio_embs)
        txt_state = self.text_projecter(txt_state.unsqueeze(1).unsqueeze(1)).repeat(1, T, 1, 1)
        vision_features, vision_pos_enc, backbone_fpn = backbone_out['vision_features'], backbone_out['vision_pos_enc'], backbone_out['backbone_fpn']

        _vision_features = vision_features.flatten(-2).permute(0, 2, 1).view(B, T, -1, self.interactor.d_model)
        omni_rep = torch.cat([_vision_features, audio_embs, txt_state], dim=-2)
        audio_embs = omni_rep[:, :, -2, :].view(B*T, 1, -1)
        B, T, N, C = omni_rep.shape
        omni_rep = omni_rep.view(B, -1, C)
        # TODO 没加interactor直接用audio embed作为prompt
        mask = self.decoder(backbone_fpn, audio_embs)


        # if self.training:
        return mask


from models.torchvggish import vggish, vggish_input


class audio_extractor(torch.nn.Module):
    def __init__(self, args, device=None):
        super(audio_extractor, self).__init__()
        self.audio_backbone = vggish.VGGish(args, device)

    def forward(self, audio_path):
        audio = vggish_input.wavfile_to_examples(audio_path)
        if audio.shape[0] != 10:
            print('lm.shape: ', audio.shape)
            N_SECONDS, CHANNEL, N_BINS, N_BANDS = audio.shape
            new_lm_tensor = torch.zeros(5, CHANNEL, N_BINS, N_BANDS)
            new_lm_tensor[:N_SECONDS] = audio
            new_lm_tensor[N_SECONDS:] = audio[-1].repeat(5 - N_SECONDS, 1, 1, 1)
            audio = new_lm_tensor
        audio_fea = self.audio_backbone(audio.to(self.audio_backbone.device))
        return audio_fea




