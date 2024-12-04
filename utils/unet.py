import torch.nn.functional as F
import torch.nn as nn

### ATTENTIONS
class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x):
        # Flatten spatial dimensions for attention
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)  # Shape: (b, hw, c)
        x = self.norm(x)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1).view(b, c, h, w)  # Reshape back
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, context_dim, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, dim)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)

    def forward(self, x, context):
        # Flatten spatial dimensions for attention
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1)  # Shape: (b, hw, c)
        x = self.norm(x)
        x, _ = self.attention(x, context, context)
        x = x.permute(0, 2, 1).view(b, c, h, w)  # Reshape back
        return x
    
### RESNET
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

### UNET BLOCK
class AttentionBlock(nn.Module):
    def __init__(self, dim, context_dim=None, heads=4, use_cross_attention=False):
        super().__init__()
        self.resnet = ResNetBlock(dim, dim)
        self.self_attention = SelfAttentionBlock(dim, heads)
        self.cross_attention = None
        if use_cross_attention:
            self.cross_attention = CrossAttentionBlock(dim, context_dim, heads)

    def forward(self, x, context=None):
        x = self.resnet(x)
        x = self.self_attention(x)
        if self.cross_attention and context is not None:
            x = self.cross_attention(x, context)
        return x
    
class UNet(nn.Module):
    def __init__(self, image_size=256, in_channels=3, base_channels=64, context_dim=512):
        super().__init__()
        self.encoder = nn.ModuleList([
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            AttentionBlock(base_channels, context_dim, use_cross_attention=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            AttentionBlock(base_channels * 2, context_dim, use_cross_attention=True)
        ])
        
        self.bottleneck = AttentionBlock(base_channels * 2, context_dim, use_cross_attention=True)

        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1),
            AttentionBlock(base_channels, context_dim, use_cross_attention=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)
        ])

    def forward(self, x, context):
        skip_connections = []
        for layer in self.encoder:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x)
            skip_connections.append(x)

        x = self.bottleneck(x, context)

        for layer in self.decoder:
            if isinstance(layer, AttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x + skip_connections.pop())
        
        return x
