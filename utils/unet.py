import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
    
    def forward(self, x):
        # x: [batch_size, num_patches, embed_dim]
        return self.attention(x, x, x)[0]

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, text_dim):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8)
    
    def forward(self, image_tokens, text_tokens):
        # Query: image_tokens, Key & Value: text_tokens
        return self.attention(image_tokens, text_tokens, text_tokens)[0]


class UNetBlock(nn.Module):
    def __init__(self, embed_dim, text_dim):
        super(UNetBlock, self).__init__()
        self.self_attention = SelfAttention(embed_dim)
        self.cross_attention = CrossAttention(embed_dim, text_dim)
        self.conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

    def forward(self, x, text_tokens):
        # x: image latent tokens, text_tokens: CLIP text embeddings
        x = self.self_attention(x)  # Self-attention
        x = self.cross_attention(x, text_tokens)  # Cross-attention
        return self.conv(x)

class DiffusionUNet(nn.Module):
    def __init__(self, embed_dim, text_dim):
        super(DiffusionUNet, self).__init__()
        self.downsample = nn.Conv2d(3, embed_dim, kernel_size=4, stride=2, padding=1)
        self.block = UNetBlock(embed_dim, text_dim)
        self.upsample = nn.ConvTranspose2d(embed_dim, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x, text_tokens):
        x = self.downsample(x)
        x = self.block(x, text_tokens)
        x = self.upsample(x)
        return x
