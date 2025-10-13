import torch
import torch.nn as nn


class Interactor(nn.Module):
    def __init__(self,
                 num_heads: int = 8,
                 d_model: int = 256,
                 num_layers :int = 6,
                 ):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.Sequential(
            *[nn.TransformerEncoderLayer(d_model, nhead=num_heads, dim_feedforward=2*d_model) for _ in range(num_layers)],
        )

    def forward(self, x):
        return self.layers(x)