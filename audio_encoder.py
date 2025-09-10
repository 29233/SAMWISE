import opts
import torch
import argparse
from models.torchvggish import vggish, vggish_input


class audio_extractor(torch.nn.Module):
    def __init__(self, args, device):
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


if __name__ == '__main__':
    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    parser = argparse.ArgumentParser('SAMWISE training and evaluation script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    name_exp = args.name_exp
    import numpy as np
    # 示例音频路径
    audio_path = "/18018998051/Ref-AVS/data/REFAVS/media/--iSerV5DbY_68000_78000/audio.wav"
    audio_encoder = audio_extractor(args, torch.device('cuda:0'))
    pytorch_output = audio_encoder(audio_path)
    # Towhee Pipeline
    from towhee import pipe, ops
    towhee_pipeline = (
        pipe.input('path')
            .map('path', 'frame', ops.audio_decode.ffmpeg())
            .map('frame', 'vecs', ops.audio_embedding.vggish())
            .output('vecs')
    )

    # 获取 Towhee 输出
    towhee_output = towhee_pipeline(audio_path).get()[0]

    # 比较输出
    print("Shape Match:", towhee_output.shape == pytorch_output.shape)
    print("Max Absolute Difference:", np.max(np.abs(towhee_output - pytorch_output.detach().cpu().numpy())))
    print("All Close (tolerance=1e-4):", np.allclose(towhee_output, pytorch_output.detach().cpu().numpy(), atol=1e-4))