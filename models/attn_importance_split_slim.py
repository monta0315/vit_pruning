import re
from audioop import bias

import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim,dim1, heads=8, dim_head=64, qkv_bias=False, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attn_attend = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)

        # self.attn_to_qkv = nn.Linear(dim, inner_dim * 3, bias = qkv_bias)
        self.attn_to_q = nn.Linear(dim, dim1, bias=qkv_bias)
        self.attn_to_k = nn.Linear(dim, dim1, bias=qkv_bias)
        self.attn_to_v = nn.Linear(dim, dim1, bias=qkv_bias)

        self.attn_to_out = (
            nn.Sequential(nn.Linear(dim1, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        q = self.attn_to_q(x)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = self.attn_to_k(x)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = self.attn_to_v(x)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attn_attend(dots)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_to_out(out)


class CompressAttention(nn.Module):
    def __init__(
        self, dim,dim1, heads=8, qkv_bias=False, dim_head=64, dropout=0.0, reduce=None
    ):
        super().__init__()
        self.heads = heads
        self.dim1 = dim1
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.num_heads = heads
        self.scale = dim_head ** -0.5
        # self.attn_to_qkv = nn.Linear(dim, (inner_dim -1 ) * 3, bias = qkv_bias)
        self.attn_to_q = nn.Linear(dim, dim1 - 1, bias=qkv_bias)
        self.attn_to_k = nn.Linear(dim, dim1 - 1, bias=qkv_bias)
        self.attn_to_v = nn.Linear(dim, dim1 - 1, bias=qkv_bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_to_out = (
            nn.Sequential(nn.Linear(dim1 - 1, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.attn_attend = nn.Softmax(dim=-1)
        self.reduce = int(reduce)

    def forward(self, x):
        B, N, _, h = *x.shape, self.heads
        q = self.attn_to_q(x)
        k = self.attn_to_k(x)
        v = self.attn_to_v(x)

        cat_tensor = torch.zeros((B, N, 1)).cuda(device=q.device)

        new_q = torch.cat(
            (q[:, :, : self.reduce], cat_tensor.detach(), q[:, :, self.reduce :]), dim=2
        )
        new_k = torch.cat(
            (k[:, :, : self.reduce], cat_tensor.detach(), k[:, :, self.reduce :]), dim=2
        )
        new_v = torch.cat(
            (v[:, :, : self.reduce], cat_tensor.detach(), v[:, :, self.reduce :]), dim=2
        )

        new_q = new_q.reshape(B, N, self.num_heads, self.dim1 // self.num_heads).permute(
            0, 2, 1, 3
        )
        new_k = new_k.reshape(B, N, self.num_heads, self.dim1 // self.num_heads).permute(
            0, 2, 1, 3
        )
        new_v = new_v.reshape(B, N, self.num_heads, self.dim1 // self.num_heads).permute(
            0, 2, 1, 3
        )

        dots = torch.matmul(new_q, new_k.transpose(-1, -2)) * self.scale

        attn = self.attn_attend(dots)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, new_v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = torch.cat((out[:, :, : self.reduce], out[:, :, self.reduce + 1 :]), dim=2)
        return self.attn_to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        reduce,
        ind,
        dropout=0.0,
        cfg=None,
        qkv_bias=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(depth):
            if i == ind:
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim,
                                CompressAttention(
                                    dim,
                                    dim1=cfg[i],
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=dropout,
                                    reduce=reduce,
                                    qkv_bias=qkv_bias,
                                ),
                            ),
                            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                        ]
                    )
                )
            else:
                self.layers.append(
                    nn.ModuleList(
                        [
                            PreNorm(
                                dim,
                                Attention(
                                    dim,
                                    dim1=cfg[i],
                                    heads=heads,
                                    dim_head=dim_head,
                                    dropout=dropout,
                                    qkv_bias=qkv_bias,
                                ),
                            ),
                            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                        ]
                    )
                )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        qkv_bias=False,
        reduce=None,
        ind=-1,
        cfg=None
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)


        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        assert(cfg != None), "cfg should not be None!"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.reduce = reduce

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            reduce=reduce,
            ind=ind,
            dropout=dropout,
            cfg=cfg,
            qkv_bias=qkv_bias,
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
