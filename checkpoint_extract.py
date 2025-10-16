import torch
import torch.nn as nn
from omegaconf import OmegaConf
from hydra.utils import instantiate

# 1. 加载SAM2预训练权重
sam2_weights = torch.load('/18018998051/SAMWISE/pretrain/sam2.1_hiera_large.pt', map_location='cpu')
config = OmegaConf.load('models/sam2/sam2_configs/base.yaml')
model = instantiate(config.model, _recursive_=True)
# 2. 查看键名，确定Image Encoder的前缀
print("SAM2权重键名示例:", list(sam2_weights.keys())[:5])

# 3. 提取Image Encoder权重
image_encoder_weights = {}
for key, value in sam2_weights['model'].items():
    if key.startswith('image_encoder.'):
        new_key = key.replace('image_encoder.', '')
        image_encoder_weights[new_key] = value

# 4. 保存提取的权重
torch.save(image_encoder_weights, 'sam2_image_encoder_weights.pth')
print(f"Image Encoder权重已保存到 sam2_image_encoder_weights.pth (数量: {len(image_encoder_weights)})")



# 加载权重
model.image_encoder.load_state_dict(torch.load('sam2_image_encoder_weights.pth'), strict=False)
print("权重加载成功！")

# 设置为评估模式
