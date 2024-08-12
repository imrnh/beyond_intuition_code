import torch
import torch.nn as nn

from vision_transformer.encoder import Block
from vision_transformer.patch_embeddings import PatchEmbed
from vision_transformer.weight_init import trunc_normal_


class VisionTransformer(nn.Module):
    """ Vision Transformer
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0., norm_layer=nn.LayerNorm,
                 global_pooling=False, dino=False):
        super().__init__()
        self.global_pooling = global_pooling
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        if global_pooling:
            self.fc_norm = norm_layer(embed_dim)
        else:
            self.norm = norm_layer(embed_dim)

        # Classifier head
        if dino:
            self.head = nn.Linear(embed_dim * 2, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.cls_gradients = None
        self.input_grad = None
        self.dino = dino

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def save_cls_gradients(self, cls_gradients):
        self.cls_gradients = cls_gradients

    def get_cls_gradients(self):
        return self.cls_gradients

    def save_input_gradients(self, input_grad):
        self.input_grad = input_grad

    def get_input_gradients(self):
        return self.input_grad

    def forward(self, x, register_hook=False):
        B = x.shape[0]
        #         if register_hook:
        #             x.register_hook(self.save_input_gradients)

        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks[:-1]:
            x = blk(x, register_hook=register_hook)

        x.register_hook(self.save_cls_gradients)

        x = self.blocks[-1](x, register_hook=register_hook)

        if self.global_pooling:
            x = x[:, 1:, :].mean(axis=1)
            x = self.fc_norm(x)
        elif self.dino:
            x = self.norm(x)
            x = torch.cat((x[:, 0].unsqueeze(-1), x[:, 1:, :].mean(axis=1).unsqueeze(-1)), dim=-1)
            x = x.reshape(x.shape[0], -1)
        else:
            x = self.norm(x)
            x = x[:, 0]

        x = self.head(x)
        return x