import torch
import torch.nn as nn
from omegaconf import OmegaConf
from hydra.utils import instantiate

# 1. 加载SAM2预训练权重
audio_weights = sam2_weights = torch.load('/18018998051/SAMWISE/pretrain/CLAP/630k-audioset-best.pt', map_location='cpu')
# sam2_weights = torch.load('/18018998051/SAMWISE/pretrain/sam2.1_hiera_large.pt', map_location='cpu')
config = OmegaConf.load('models/config/SCLAP_full.yaml')
model = instantiate(config.model, _recursive_=True)
# 2. 查看键名，确定Image Encoder的前缀
print("SAM2权重键名示例:", list(sam2_weights.keys()))

# 3. 提取Image Encoder权重
image_encoder_weights = {}
for key, value in sam2_weights['state_dict'].items():
    if key.startswith('module.text_branch'):
        new_key = key.replace('module.text_branch.', '')
        image_encoder_weights[new_key] = value

# 4. 保存提取的权重
torch.save(image_encoder_weights, '/18018998051/SAMWISE/pretrain/CLAP/CLAP_text_encoder.pth')
print(f"Image Encoder权重已保存到 /18018998051/SAMWISE/pretrain/CLAP/CLAP_text_encoder.pth (数量: {len(image_encoder_weights)})")



# 加载权重
model.text_encoder.load_state_dict(torch.load('/18018998051/SAMWISE/pretrain/CLAP/CLAP_text_encoder.pth'), strict=False)
print("权重加载成功！")

# 设置为评估模式
