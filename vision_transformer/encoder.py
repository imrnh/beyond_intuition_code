import torch.nn as nn
from vision_transformer.multilayer_perceptron import Mlp
from vision_transformer.attention import Attention
import vision_transformer.layers as cusl
from vision_transformer.attention_rel import AttentionWRel

"""
     Encoder without relevance.
"""


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)  # hidden_features is the output layer's neuron count.

        self.input = None
        self.output = None
        self.tilde = None

    def save_input(self, z):
        self.input = z

    def get_input(self):
        return self.input

    def save_tilde(self, z):
        self.tilde = z

    def get_tilde(self):
        return self.tilde

    def save_output(self, z):
        self.output = z

    def get_output(self):
        return self.output

    def forward(self, x, register_hook=False):
        self.save_input(x)
        out = x + self.attn(self.norm1(x), register_hook=register_hook)
        self.save_tilde(out)
        out = out + self.mlp(self.norm2(out))
        self.save_output(out)

        return out


"""
    Encoder with relevance.
"""


class BlockWRel(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = cusl.LayerNorm(dim, eps=1e-6)
        self.attn = AttentionWRel(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = cusl.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.add1 = cusl.Add()
        self.add2 = cusl.Add()
        self.clone1 = cusl.Clone()
        self.clone2 = cusl.Clone()

    def forward(self, x):
        x1, x2 = self.clone1(x, 2)
        x = self.add1([x1, self.attn(self.norm1(x2))])
        x1, x2 = self.clone2(x, 2)
        x = self.add2([x1, self.mlp(self.norm2(x2))])
        return x

    def relprop(self, cam, **kwargs):
        (cam1, cam2) = self.add2.relprop(cam, **kwargs)
        cam2 = self.mlp.relprop(cam2, **kwargs)
        cam2 = self.norm2.relprop(cam2, **kwargs)
        cam = self.clone2.relprop((cam1, cam2), **kwargs)

        (cam1, cam2) = self.add1.relprop(cam, **kwargs)
        cam2 = self.attn.relprop(cam2, **kwargs)
        cam2 = self.norm1.relprop(cam2, **kwargs)
        cam = self.clone1.relprop((cam1, cam2), **kwargs)
        return cam
