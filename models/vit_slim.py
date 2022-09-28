import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

MIN_NUM_PATCHES = 16

defaultcfg = {
    # 6 : [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
    # 6 : [[510, 375, 512, 443], [512, 399, 479, 286], [511, 367, 370, 196], [512, 404, 111, 95], [512, 425, 60, 66], [512, 365, 356, 223]]
    6 : [[360, 512], [408, 479], [360, 370], [408, 111], [432, 60], [360, 356]]
}

# classes

class channel_selection(nn.Module):
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (B, num_patches + 1, dim). 
        """
        output = input_tensor.mul(self.indexes)
        return output

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
        self.net1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        #self.select1 = channel_selection(dim)
    def forward(self, x):
        x = self.net1(x)
        #x = self.select1(x)
        return self.net2(x)

class Attention(nn.Module):
    def __init__(self, dim, dim1, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim1 ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)
    


        #self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        """
            刈り込みのためにまとめていたqkvを分割する
        """
        self.to_q = nn.Linear(dim, dim1, bias = False)
        self.to_k = nn.Linear(dim, dim1, bias = False)
        self.to_v = nn.Linear(dim, dim1, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(dim1, dim),
            nn.Dropout(dropout)
        ) 

        #self.select1 = channel_selection(dim)


    def forward(self, x,mask = None):
        #qkv = self.to_qkv(x).chunk(3, dim = -1)
        #q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        h = self.heads

        q = self.to_q(x)
        #q = self.select1(q)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k = self.to_k(x)
        #k = self.select1(k)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        v = self.to_v(x)
        #v = self.select1(v)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout,cfg):
        super().__init__()
        self.layers = nn.ModuleList([])

        if cfg is not None:
            for num in cfg:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, num[0], heads = heads, dropout = dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, num[1], dropout = dropout)))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim,dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                    Residual(PreNorm(dim, FeedForward(dim,dim, mlp_dim, dropout = dropout)))
                ]))
    def forward(self, x,mask = None):
        for attn, ff in self.layers:
            x = attn(x,mask=mask)
            x = ff(x)
        return x

class ViT_slim(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,cfg=None, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,cfg)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
