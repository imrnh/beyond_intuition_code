import torch
import torch.nn as nn
import numpy as np

import vision_transformer.layers as cusl
from vision_transformer.multilayer_perceptron import Mlp
from vision_transformer.patch_embeddings import PatchEmbedWRel
from vision_transformer.encoder import BlockWRel
from vision_transformer.weight_init import trunc_normal_
from vision_transformer.incorporate_relevance.rollout_attn_calc import compute_rollout_attention


class VisionTransformerWRel(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, mlp_head=False, drop_rate=0., attn_drop_rate=0., mae=False,
                 dino=False):
        super().__init__()
        self.mae = mae
        self.dino = dino
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbedWRel(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList([
            BlockWRel(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate)
            for i in range(depth)])

        if mae:
            self.fc_norm = cusl.LayerNorm(embed_dim)
        else:
            self.norm = cusl.LayerNorm(embed_dim)
        if mlp_head:
            # paper diagram suggests 'MLP head', but results in 4M extra parameters vs paper
            self.head = Mlp(embed_dim, int(embed_dim * mlp_ratio), num_classes)
        elif dino:
            self.head = cusl.Linear(embed_dim * 2, num_classes)
        else:
            # with a single Linear layer as head, the param count within rounding of paper
            self.head = cusl.Linear(embed_dim, num_classes)

        # FIXME not quite sure what the proper weight init is supposed to be,
        # normal / trunc normal w/ std == .02 similar to other Bert like transformers
        trunc_normal_(self.pos_embed, std=.02)  # embeddings same as weights?
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        if dino:
            self.pool1 = cusl.IndexSelect()
            self.pool2 = cusl.IndexSelect()
        else:
            self.pool = cusl.IndexSelect()
        self.add = cusl.Add()

        self.inp_grad = None

    def save_inp_grad(self, grad):
        self.inp_grad = grad

    def get_inp_grad(self):
        return self.inp_grad

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.add([x, self.pos_embed])

        x.register_hook(self.save_inp_grad)

        for blk in self.blocks:
            x = blk(x)

        if self.mae:
            x = self.fc_norm(x)
            x = self.pool(x, dim=1, indices=torch.tensor(np.arange(1, x.shape[1]), device=x.device)).mean(1)
        elif self.dino:
            x = self.norm(x)
            x = torch.cat((self.pool1(x, dim=1, indices=torch.tensor(0, device=x.device)).squeeze(1).unsqueeze(-1),
                           self.pool2(x, dim=1, indices=torch.tensor(np.arange(1, x.shape[1]), device=x.device)).mean(
                               1).unsqueeze(-1)), dim=-1)
            x = x.reshape(x.shape[0], -1)
        else:
            x = self.norm(x)
            x = self.pool(x, dim=1, indices=torch.tensor(0, device=x.device)).squeeze(1)
        x = self.head(x)
        return x

    def relprop(self, cam=None, method="transformer_attribution", is_ablation=False, start_layer=0, **kwargs):
        # print(kwargs)
        # print("conservation 1", cam.sum())
        cam = self.head.relprop(cam, **kwargs)
        cam = cam.unsqueeze(1)
        if self.dino:
            cam1 = self.pool1.relprop(cam[:, :, :768], **kwargs)
            cam2 = self.pool2.relprop(cam[:, :, 768:], **kwargs)
            cam = cam1 + cam2
        else:
            cam = self.pool.relprop(cam, **kwargs)
        if self.mae:
            cam = self.fc_norm.relprop(cam, **kwargs)
        else:
            cam = self.norm.relprop(cam, **kwargs)
        for blk in reversed(self.blocks):
            cam = blk.relprop(cam, **kwargs)

        # print("conservation 2", cam.sum())
        # print("min", cam.min())

        if method == "full":
            (cam, _) = self.add.relprop(cam, **kwargs)
            cam = cam[:, 1:]
            cam = self.patch_embed.relprop(cam, **kwargs)
            # sum on channels
            cam = cam.sum(dim=1)
            return cam

        elif method == "rollout":
            # cam rollout
            attn_cams = []
            for blk in self.blocks:
                attn_heads = blk.attn.get_attn_cam().clamp(min=0)
                avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
                attn_cams.append(avg_heads)
            cam = compute_rollout_attention(attn_cams, start_layer=start_layer)
            cam = cam[:, 0, 1:]
            return cam

        # our method, method name grad is legacy
        elif method == "transformer_attribution" or method == "grad":
            cams = []
            for blk in self.blocks:
                grad = blk.attn.get_attn_gradients()
                cam = blk.attn.get_attn_cam()
                cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
                cam = cam.clamp(min=0).mean(dim=0)
                cams.append(cam.unsqueeze(0))
            rollout = compute_rollout_attention(cams, start_layer=start_layer)
            if self.mae:
                cam = rollout[:, 1:, 1:].mean(1)
            elif self.dino:
                cam = rollout[:, 1:, 1:].mean(1) + rollout[:, 0, 1:]
            else:
                cam = rollout[:, 0, 1:]
            return cam

        elif method == "last_layer":
            cam = self.blocks[-1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[-1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            if self.mae:
                cam = cam[1:, 1:].mean(0)
            elif self.dino:
                cam = cam[1:, 1:].mean(0) + cam[0, 1:]
            else:
                cam = cam[0, 1:]
            return cam

        elif method == "last_layer_attn":
            cam = self.blocks[-1].attn.get_attn()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

        elif method == "second_layer":
            cam = self.blocks[1].attn.get_attn_cam()
            cam = cam[0].reshape(-1, cam.shape[-1], cam.shape[-1])
            if is_ablation:
                grad = self.blocks[1].attn.get_attn_gradients()
                grad = grad[0].reshape(-1, grad.shape[-1], grad.shape[-1])
                cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            cam = cam[0, 1:]
            return cam

