import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    """
    SiLU is used instead of ReLU because SiLU provides smoother activation curve compared to ReLU 
    which prevents dying of neurons and lead to better performance.
    """
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x + residual
