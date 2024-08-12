def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_config = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='models/pretrained_model/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='models/pretrained_model/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_224_moco': _cfg(
        url='models/pretrained_model/linear-vit-b-300ep.pth.tar',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_224_dino': _cfg(
        url_backbone='models/pretrained_model/dino_vitbase16_pretrain.pth',
        url_linear='models/pretrained_model/dino_vitbase16_linearweights.pth',
        url='models/pretrained_model/dino_vitbase16_pretrain_full_checkpoint.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_mae_patch16_224': _cfg(
        url='models/pretrained_model/mae_finetuned_vit_base.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_large_patch16_224': _cfg(
        url='models/pretrained_model/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

cls = ['airplane',
       'bicycle',
       'bird',
       'boat',
       'bottle',
       'bus',
       'car',
       'cat',
       'chair',
       'cow',
       'dining table',
       'dog',
       'horse',
       'motobike',
       'person',
       'potted plant',
       'sheep',
       'sofa',
       'train',
       'tv'
       ]
