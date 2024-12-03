import torch.nn as nn

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