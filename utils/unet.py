from layers.resnet import ResNetBlock
from layers.sattn import SelfAttentionBlock
from layers.cattn import CrossAttentionBlock
import torch.nn as nn

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
