import torch.nn as nn
from functools import partial

from utils.config import default_config
from utils.load_pretrained import load_pretrained
from vision_transformer import VisionTransformer, VisionTransformerWRel


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


def vit_small_patch16_224(pretrained=False, model_path=None, num_classes=1000, w_rel=False, **kwargs):
    if w_rel:
        model = VisionTransformerWRel(patch_size=16, num_classes=num_classes, embed_dim=768, depth=8, num_heads=6, mlp_ratio=3, qkv_bias=False,
                                      **kwargs)
    else:
        model = VisionTransformer(patch_size=16, num_classes=num_classes, embed_dim=768, depth=8, num_heads=6, mlp_ratio=3, qkv_bias=False,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    if model_path is not None:
        default_config['url'] = model_path

    model.default_cfg = default_config['vit_small_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


def vit_base_patch16_224(pretrained=False, model_path=None, num_classes=1000, w_rel=False, **kwargs):
    if w_rel:
        model = VisionTransformerWRel(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                      **kwargs)
    else:
        model = VisionTransformer(patch_size=16, num_classes=num_classes, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_config['vit_base_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model


def vit_base_patch16_224_dino(pretrained=False, w_rel=False, **kwargs):
    if w_rel:
        model = VisionTransformerWRel(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                      dino=True, **kwargs)
    else:
        model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), dino=True, **kwargs)

    model.default_cfg = default_config['vit_base_patch16_224_dino']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, dino=True)
    return model


def vit_base_patch16_224_moco(pretrained=False, w_rel=False, **kwargs):
    if w_rel:
        model = VisionTransformerWRel(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                      **kwargs)
    else:
        model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    model.default_cfg = default_config['vit_base_patch16_224_moco']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, moco=True)
    return model


def vit_mae_patch16_224(pretrained=False, w_rel=False, **kwargs):
    if w_rel:
        model = VisionTransformerWRel(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                      mae=True, **kwargs)
    else:
        model = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), global_pooling=True, **kwargs)
    model.default_cfg = default_config['vit_mae_patch16_224']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, mae=True)
    return model


def vit_large_patch16_224(pretrained=False, w_rel=False, **kwargs):
    if w_rel:
        model = VisionTransformerWRel(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                                      **kwargs)
    else:
        model = VisionTransformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
                                  norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = default_config['vit_large_patch16_224']
    if pretrained:
        load_pretrained(model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
