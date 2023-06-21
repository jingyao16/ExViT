import torch
import torch.nn as nn
from einops import rearrange

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x, mask = None):
        # x:[b,n,dim]
        _, _, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        # head_num*head_dim = dim
        qkv = self.to_qkv(x).chunk(3, dim = -1) # split to_qkv(x) from [b, n, inner_dim*3] to [b, n, inner_dim]*3 tuple
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1) # [b,head_num,n,n]
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v) # [b,head_num,n,head_dim]
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, mode):
        super().__init__()
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))

        self.mode = mode

    def forward(self, x, mask = None):
        if self.mode == 'MViT':
            for attn, ff in self.layers:
                x = attn(x, mask = mask)
                x = ff(x)
        return x

class MViT(nn.Module):
    def __init__(self, patch_size, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head = 16, dropout=0., emb_dropout=0., mode='MViT'):
        super().__init__()

        nout = 16
        
        samesize = 1
 
        self.separable1 = nn.Sequential(
            nn.Conv2d(num_patches[0], num_patches[0], kernel_size=3, padding=samesize, groups=num_patches[0]),
            nn.Conv2d(num_patches[0], nout, kernel_size=1),
            nn.BatchNorm2d(nout),
            nn.GELU(),
            nn.Conv2d(nout, nout, kernel_size=3, padding=samesize, groups=nout),
            nn.Conv2d(nout, nout*2, kernel_size=1),
            nn.BatchNorm2d(nout*2),
            nn.GELU(),
            nn.Conv2d(nout*2, nout*2, kernel_size=3, padding=samesize, groups=nout*2),
            nn.Conv2d(nout*2, nout*4, kernel_size=1),
            nn.BatchNorm2d(nout*4),
            nn.GELU()
        )
        
        self.separable2 = nn.Sequential(
            nn.Conv2d(num_patches[1], num_patches[1], kernel_size=3, padding=samesize, groups=num_patches[1]),
            nn.Conv2d(num_patches[1], nout, kernel_size=1),
            nn.BatchNorm2d(nout),
            nn.GELU(),
            nn.Conv2d(nout, nout, kernel_size=3, padding=samesize, groups=nout),
            nn.Conv2d(nout, nout*2, kernel_size=1),
            nn.BatchNorm2d(nout*2),
            nn.GELU(),
            nn.Conv2d(nout*2, nout*2, kernel_size=3, padding=samesize, groups=nout*2),
            nn.Conv2d(nout*2, nout*4, kernel_size=1),
            nn.BatchNorm2d(nout*4),
            nn.GELU()
        )
        
        grid_size = 1
        vit_patches = (patch_size // grid_size) ** 2
        self.to_patch_embedding2 = nn.Linear(nout*4, dim)
        self.to_patch_embedding2c = nn.Linear(nout*4, dim)
        
        self.pos_embedding = nn.Parameter(torch.randn(1, vit_patches, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth-4, heads, dim_head, mlp_dim, dropout, mode)
        self.transformer1 = Transformer(dim, depth-2, heads, dim_head, mlp_dim, dropout, mode)
        self.transformer2 = Transformer(dim, depth-2, heads, dim_head, mlp_dim, dropout, mode)
        
        self.mlp_head0 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.Softmax(dim=1)
            )

        self.mlp_head1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
            )
        
    def forward(self, x1, x2, mask = None):
        
        # Multimodal Feature Extraction & Tokenization
        x1 = self.separable1(x1)
        x1 = rearrange(x1, 'b c h w -> b (h w) c')
        x1 = self.to_patch_embedding2(x1) #[b, n, c to dim], n=hw
        b, n, _ = x1.shape
        x1 += self.pos_embedding[:, :n]
        x1 = self.dropout(x1)
        
        x2 = self.separable2(x2)
        x2 = rearrange(x2, 'b c h w -> b (h w) c')
        x2 = self.to_patch_embedding2c(x2) #common subpsace projection: better to be different with self.to_patch_embedding1
        x2 += self.pos_embedding[:, :n]
        x2 = self.dropout(x2)
        
        # Attention Fusion
        x1 = self.transformer1(x1)
        x2 = self.transformer2(x2)
        
        x = torch.cat((x1,x2), dim=1)
        x = self.transformer(x)
        
        # MLP Pre-Head & Head
        xs = torch.squeeze(self.mlp_head0(x))#b-n
        x = torch.einsum('bn,bnd->bd', xs, x)
        
        x = self.mlp_head1(x)

        return x