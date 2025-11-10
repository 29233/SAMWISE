import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from training.data import get_audio_features, int16_to_float32, float32_to_int16
from clap_module.htsat import HTSAT_Swin_Transformer


class AudioEncoder(nn.Module):

    def __init__(self, encoder_model):
        super().__init__()
        self.audio_encoder = encoder_model

    def load_from(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        missing_keys, unexpected_keys = self.audio_encoder.load_state_dict(checkpoint, strict=False)
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    def encode_audio(self, audio, device, audio_inter=False):
        return self.audio_encoder(audio, mixup_lambda=None, device=device, audio_inter=audio_inter)

    def get_audio_embedding(self, data, audio_inter=False):
        """Get the audio embedding from the model

        Parameters
        ----------
        data: a list of dict
            the audio input dict list from 'get_audio_feature' method

        Returns
        ----------
        audio_embed: torch.Tensor
            a tensor of audio_embeds (N, D)

        """
        device = next(self.parameters()).device
        input_dict = {}
        keys = data[0].keys()
        for k in keys:
            input_dict[k] = torch.cat([d[k].unsqueeze(0) for d in data], dim=0).to(device)
        audio_outs = self.encode_audio(input_dict, device=device, audio_inter=audio_inter)
        audio_embeds = audio_outs["embedding"]
        if audio_inter:
            inter_embeds = audio_outs["audio_inter"]
            inter_embeds = F.normalize(inter_embeds, dim=-1)
        else:
            inter_embeds = None
        audio_embeds = F.normalize(audio_embeds, dim=-1)
        return audio_embeds, inter_embeds

    def forward(self, x, use_tensor=True, audio_inter=False):
        self.audio_encoder.eval()
        audio_input = []
        bs = len(x)
        for f in x:
            # load the waveform of the shape (T,), should resample to 48000
            audio_waveform, _ = librosa.load(f, sr=48000)
            # quantize
            audio_waveform = int16_to_float32(float32_to_int16(audio_waveform))
            audio_waveform = torch.from_numpy(audio_waveform).float()
            temp_dict = {}
            temp_dict = get_audio_features(
                temp_dict, audio_waveform, 480000,
                data_truncating='fusion' if self.audio_encoder.enable_fusion else 'rand_trunc',
                data_filling='repeatpad',
                audio_cfg=self.audio_encoder.config,
                require_grad=audio_waveform.requires_grad
            )
            audio_input.append(temp_dict)
        audio_embed, inter_embed = self.get_audio_embedding(audio_input, audio_inter)
        audio_embed = audio_embed.view(bs, 10, 768)
        if audio_inter:
            inter_embed = inter_embed.view(bs, 10, 768)
        if not use_tensor:
            audio_embed = audio_embed.detach().cpu().numpy()
        return audio_embed, inter_embed