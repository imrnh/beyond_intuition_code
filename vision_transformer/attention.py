import torch
import torch.nn as nn
from einops import rearrange


# noinspection PyTupleAssignmentBalance
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5  # This is root_over(dh) with which we divided Q.KT

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)  # Attention's multiplication. That is, we have to pass the attention output through a Linear layer. That is the linear layer.
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_gradients = None
        self.attention_map = None

        self.vproj = None
        self.input = None

    def save_input(self, z):
        self.input = z

    def get_input(self):
        return self.input

    # Backward hook. Therefore, param is automatically gradients for that layer.
    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def forward(self, x, register_hook=False):
        b, n, _, h = *x.shape, self.num_heads
        self.save_input(x)

        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # Simple torch.matmul actually.

        self.save_attention_map(attn)
        if register_hook:
            attn.register_hook(self.save_attn_gradients)

        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.proj(out)
        out = self.proj_drop(out)


        self.vproj = torch.matmul(rearrange(v, 'b h n d -> b n (h d)'), self.proj.weight.t())

        return out
