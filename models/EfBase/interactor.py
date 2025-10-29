import torch
import torch.nn as nn
import random


class Interactor(nn.Module):
    def __init__(self,
                 num_heads: int = 8,
                 d_model: int = 256,
                 num_layers: int = 6,
                 shuffle: bool = False,
                 ):
        super().__init__()
        self.d_model = d_model
        self.shuffle = shuffle
        self.layers = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model, nhead=num_heads, dim_feedforward=2*d_model) for _ in range(num_layers)],
        )

    @staticmethod
    def shuffle_tensor_groups(tensor, num_groups):
        """
        将形状为 (batch, seq_len, feature) 的张量按 seq_len 划分为 10 组（每组 146 个元素），
        然后对这 10 组进行随机打乱（以组为单位）。
        参数:
            tensor: 输入张量，形状应为 (20, 1460, 256)
        返回:
            shuffle_tensor: 打乱后的张量，形状仍为 (20, 1460, 256)
        """
        # 生成随机索引（0 到 9 的随机排列）
        indices = list(range(num_groups))
        random.shuffle(indices)
        # 按随机索引重新组合组
        shuffled_groups = [tensor[:, i] for i in indices]
        # 拼接回原始形状
        shuffled_tensor = torch.cat(shuffled_groups, dim=1)
        return shuffled_tensor

    def forward(self, x):
        B, T, N, C = x.shape
        if self.shuffle:
            x = self.shuffle_tensor_groups(x, T)
        x = x.view(B, -1, C)
        x = self.layers(x).view(B * T, -1, C)
        return x


class Interactor_with_mask(nn.Module):
    def __init__(self,
                 num_heads: int = 8,
                 d_model: int = 256,
                 num_layers: int = 6,
                 shuffle: bool = False,
                 ):
        super().__init__()
        self.d_model = d_model
        self.shuffle = shuffle
        self.layers = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model, nhead=num_heads, dim_feedforward=2*d_model) for _ in range(num_layers)],
        )

    @staticmethod
    def shuffle_tensor_groups(tensor, num_groups):
        """
        将形状为 (batch, seq_len, feature) 的张量按 seq_len 划分为 10 组（每组 146 个元素），
        然后对这 10 组进行随机打乱（以组为单位）。
        参数:
            tensor: 输入张量，形状应为 (20, 1460, 256)
        返回:
            shuffle_tensor: 打乱后的张量，形状仍为 (20, 1460, 256)
        """
        # 生成随机索引（0 到 9 的随机排列）
        indices = list(range(num_groups))
        random.shuffle(indices)
        # 按随机索引重新组合组
        shuffled_groups = [tensor[:, i] for i in indices]
        # 拼接回原始形状
        shuffled_tensor = torch.cat(shuffled_groups, dim=1)
        return shuffled_tensor

    def forward_layers(self, x, attention_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=attention_mask)
        return x

    @staticmethod
    def process_mask(B, N, attention_mask):
        visual_mask = torch.ones(B, N - attention_mask.shape[-1], dtype=attention_mask.dtype, device=attention_mask.device)
        full_mask = torch.cat([visual_mask, attention_mask], dim=1)
        return full_mask


    def forward(self, x, attention_mask=None):
        B, T, N, C = x.shape
        if self.shuffle:
            x = self.shuffle_tensor_groups(x, T)
        x = x.view(B, -1, C)
        if attention_mask is not None:
            attention_mask = attention_mask.bool().cuda()
            full_mask = self.process_mask(B, N, attention_mask)
            final_mask = full_mask.repeat(1, T)
            x = self.forward_layers(x, final_mask)
        else:
            x = self.layers(x).view(B * T, -1, C)
        return x


class DummyInteractor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, T, N, C = x.shape
        x = x.view(B*T, -1, C)
        return x, None