import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    """Pre-normalization layer"""
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """Feed-forward network"""
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Multi-head self-attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TimeSformerBlock(nn.Module):
    """
    TimeSformer block with divided space-time attention.
    
    순서:
    1. Temporal attention (시간축)
    2. Spatial attention (공간축)
    3. Feed-forward
    """
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout, num_patches, num_frames):
        super().__init__()
        self.num_patches = num_patches
        self.num_frames = num_frames
        
        # Temporal attention
        self.temporal_attn = PreNorm(dim, Attention(dim, heads, dim_head, dropout))
        
        # Spatial attention
        self.spatial_attn = PreNorm(dim, Attention(dim, heads, dim_head, dropout))
        
        # Feed-forward
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_frames * num_patches + 1, dim)
               +1 for cls token
        """
        b = x.shape[0]
        cls_token = x[:, :1]  # (b, 1, dim)
        x = x[:, 1:]  # (b, num_frames * num_patches, dim)
        
        # Reshape for temporal attention: group by spatial position
        # (b, num_frames * num_patches, dim) -> (b * num_patches, num_frames, dim)
        x = rearrange(x, 'b (t p) d -> (b p) t d', t=self.num_frames, p=self.num_patches)
        x = x + self.temporal_attn(x)
        x = rearrange(x, '(b p) t d -> b (t p) d', b=b, p=self.num_patches)
        
        # Reshape for spatial attention: group by time
        # (b, num_frames * num_patches, dim) -> (b * num_frames, num_patches, dim)
        x = rearrange(x, 'b (t p) d -> (b t) p d', t=self.num_frames, p=self.num_patches)
        x = x + self.spatial_attn(x)
        x = rearrange(x, '(b t) p d -> b (t p) d', b=b, t=self.num_frames)
        
        # Combine with cls token and apply feed-forward
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.ff(x)
        
        return x


class TimeSformer(nn.Module):
    """
    TimeSformer: Is Space-Time Attention All You Need for Video Understanding?
    
    Divided space-time attention 방식 구현
    - 먼저 시간축 attention
    - 그 다음 공간축 attention
    """
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_frames=8,
        num_classes=8,  # 8개 손가락 각도
        dim=768,
        depth=12,
        heads=12,
        mlp_dim=3072,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1
    ):
        super().__init__()
        
        assert image_size % patch_size == 0, 'Image size must be divisible by patch size'
        
        self.patch_size = patch_size
        self.num_frames = num_frames
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches
        patch_dim = 3 * patch_size ** 2
        
        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )
        
        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.temporal_embedding = nn.Parameter(torch.randn(1, num_frames, dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer blocks
        self.transformer = nn.ModuleList([
            TimeSformerBlock(dim, heads, dim_head, mlp_dim, dropout, num_patches, num_frames)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        
        # Regression head
        self.head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, num_frames, channels, height, width) or
               (batch, channels, num_frames, height, width)
        
        Returns:
            (batch, num_classes)
        """
        # Handle input format
        if x.dim() == 5:
            b, dim1, dim2, h, w = x.shape
            if dim2 == 3 and dim1 != 3:
                # (B, T, C, H, W) -> keep as is
                pass
            elif dim1 == 3:
                # (B, C, T, H, W) -> (B, T, C, H, W)
                x = x.permute(0, 2, 1, 3, 4)
        
        b, t, c, h, w = x.shape
        
        # Patch embedding: (b, t, num_patches, dim)
        x = self.to_patch_embedding(x)
        
        # Add positional embedding to each frame
        x = x + self.pos_embedding[:, 1:, :]  # Skip cls token position
        
        # Add temporal embedding
        temporal_emb = self.temporal_embedding[:, :t, :].unsqueeze(2)  # (1, t, 1, dim)
        x = x + temporal_emb
        
        # Flatten: (b, t * num_patches, dim)
        x = rearrange(x, 'b t p d -> b (t p) d')
        
        # Prepend cls token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.transformer:
            x = block(x)
        
        x = self.norm(x)
        
        # Use cls token for prediction
        cls_output = x[:, 0]
        
        return self.head(cls_output)


class TimeSformerSmall(TimeSformer):
    """작은 버전의 TimeSformer - 데이터가 적을 때 사용"""
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_frames=8,
        num_classes=8,
        dropout=0.1
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            num_classes=num_classes,
            dim=384,        # 768 -> 384
            depth=6,        # 12 -> 6
            heads=6,        # 12 -> 6
            mlp_dim=1536,   # 3072 -> 1536
            dim_head=64,
            dropout=dropout,
            emb_dropout=dropout
        )


class TimeSformerTiny(TimeSformer):
    """아주 작은 버전 - 빠른 실험용"""
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_frames=8,
        num_classes=8,
        dropout=0.2
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            num_classes=num_classes,
            dim=192,        # 더 작게
            depth=4,        # 4 layers
            heads=3,
            mlp_dim=768,
            dim_head=64,
            dropout=dropout,
            emb_dropout=dropout
        )


def create_model(config):
    """
    Create model from config dictionary.
    """
    model_type = config.get('model_type', 'timesformer_small')
    num_frames = config.get('num_frames', 8)
    image_size = config.get('image_size', 224)
    patch_size = config.get('patch_size', 16)
    num_outputs = config.get('num_outputs', 8)
    dropout = config.get('dropout', 0.1)
    
    if model_type == 'timesformer':
        model = TimeSformer(
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            num_classes=num_outputs,
            dropout=dropout
        )
    elif model_type == 'timesformer_small':
        model = TimeSformerSmall(
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            num_classes=num_outputs,
            dropout=dropout
        )
    elif model_type == 'timesformer_tiny':
        model = TimeSformerTiny(
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            num_classes=num_outputs,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == '__main__':
    # Test models
    print("Testing TimeSformer variants...")
    
    # Test input: (batch=2, frames=8, channels=3, H=224, W=224)
    test_input = torch.randn(2, 8, 3, 224, 224)
    
    for name, Model in [('Tiny', TimeSformerTiny), ('Small', TimeSformerSmall)]:
        model = Model(num_frames=8)
        output = model(test_input)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n{name}:")
        print(f"  Input shape: {test_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Parameters: {total_params:,}")
