import torch.nn as nn
import torch
import torch.nn.functional as F

from einops import rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, mask = None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_mask = mask.bool()
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0
        )
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
class InstillAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., cfg=None):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        
        feature_dim = cfg.feature_dim if cfg.different_learnable_tokens else dim
        self.to_anotherv = nn.Linear(feature_dim, inner_dim, bias = False)
        
        self.to_yout = nn.Sequential(
            nn.Linear(inner_dim, feature_dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        

    def forward(self, x, y, mask = None):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        another_v = self.to_anotherv(y)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        another_v = rearrange(another_v, 'b n (h d) -> b h n d', h = self.heads)
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            attn_mask = mask.bool()
        else:
            attn_mask = None

        x_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0
        )
        x_out = rearrange(x_out, 'b h n d -> b n (h d)')
        x_out = self.to_out(x_out)
        
        y_out = F.scaled_dot_product_attention(
            q, k, another_v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0
        )
        y_out = rearrange(y_out, 'b h n d -> b n (h d)')
        y_out = self.to_yout(y_out)
        
        return x_out, y_out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., cfg=None):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.cfg = cfg
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, mask = None, context_feature=None):
        for attn, ff in self.layers:
            x = attn(x, mask) + x
            x = ff(x) + x
        return self.norm(x)
    
class InstillTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., cfg=None):
        super().__init__()
        feature_mlp_dim = cfg.feature_dim * 2 if cfg.different_learnable_tokens else mlp_dim
        feature_dim = cfg.feature_dim if cfg.different_learnable_tokens else dim
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        self.cfg = cfg
        self.y_norm = nn.LayerNorm(feature_dim)
        for _ in range(depth):                
            self.layers.append(nn.ModuleList([
                InstillAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout, cfg=cfg),
                FeedForward(dim, mlp_dim, dropout = dropout),
                FeedForward(feature_dim, feature_mlp_dim, dropout = dropout),
            ]))

    def forward(self, x, mask = None, context_feature=None):
        for attn, ff1, ff2 in self.layers:
            
            x_attn, y_attn = attn(x, context_feature, mask)
            x = x_attn + x
            context_feature = y_attn + context_feature
            x = ff1(x) + x
            context_feature = ff2(context_feature) + context_feature
        return self.norm(x), self.y_norm(context_feature)