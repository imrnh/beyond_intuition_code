import os


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


# Check if models/pretrained_model exists or not.
# If the directory doesn't exists, the reason might be cause we are using kaggle notebook.
# Therefore, load using kaggle.
# if os.path.exists("models/pretrained_model"):
#     model_base_dir = "models/pretrained_model"
# else:
#     model_base_dir = "/kaggle/input/popular-vits-for-interpretation/pytorch/default/1"

# print(f"Model base directory set to: {model_base_dir}. @utils.config.py.22")


model_base_dir = "models/pretrained_model"


default_config = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url=f'{model_base_dir}/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url=f'{model_base_dir}/jx_vit_base_p16_224-80ecf9dd.pth',
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
