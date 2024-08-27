import os


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


model_base_dir = "lib/pretrained_model"


default_config = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url=f'{model_base_dir}/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url=f'{model_base_dir}/jx_vit_base_p16_224-80ecf9dd.pth',
        # url="C:/Users/muimr/Research/Vit Interpret/Codes/beyond_intuition/lib/benchmark__trained_on_noisy_data/ff.pth",
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_224_moco': _cfg(
        url=f'{model_base_dir}/linear-vit-b-300ep.pth.tar',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_224_dino': _cfg(
        url_backbone=f'{model_base_dir}/dino_vitbase16_pretrain.pth',
        url_linear=f'{model_base_dir}/dino_vitbase16_linearweights.pth',
        url=f'{model_base_dir}/dino_vitbase16_pretrain_full_checkpoint.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_mae_patch16_224': _cfg(
        url=f'{model_base_dir}/mae_finetuned_vit_base.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_large_patch16_224': _cfg(
        url=f'{model_base_dir}/jx_vit_large_p16_224-4ee7a4dc.pth',
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
