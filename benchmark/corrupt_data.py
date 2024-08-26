"""
    For a given dataset, corrupt each image using saliency map.

"""
import torch
import torch.nn as nn
import numpy as np

def get_all_channels_mean(image):
    channel_mean = []
    permuted_image = torch.permute(torch.Tensor(image), (2,1,0))
    for ch in range(image.shape[-1]):
        channel_mean.append(permuted_image[ch].mean())

    return channel_mean


def make_patches(images, n_patches):
    """

    :param images: take a batch of input images
    :param n_patches: number of patches per dimension (width, height). Total patch = n_patches * n_patches
    :return: Array of 1D patch embeddings.
    """

    n, c, h, w = images.shape  # n = batch_size, c = num_channels
    patch_size = h // n_patches  # As h == w, no need for w.
    patch_dimension = (h * w) // (n_patches * n_patches)

    make_flatten = nn.Flatten()


    assert h == w, "Patch embeddings require both height and weight to be equal"

    patches = torch.zeros((n, n_patches * n_patches, c, patch_dimension))  # 2nd param = patch count per image = n_patches * n_patches. 3rd param to store the patches.

    for image_idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size,
                        j * patch_size: (j + 1) * patch_size]  # First indexing done for channel.
                patches[image_idx, i * n_patches + j] = make_flatten(patch)

    return patches


def get_corrupted_image(image, saliency, corrupt_percentage=None, corrupt_count=None):
    saliency = torch.Tensor(saliency)
    saliency_flat = torch.flatten(saliency)
    saliency_sorted, sorted_indices = torch.sort(saliency_flat, descending=True)

    if corrupt_percentage is not None:
      top_k = int(corrupt_percentage * len(saliency_sorted))  # number of classes to be sorted.
    else:
      top_k = corrupt_count

    channel_mean = get_all_channels_mean(image)

    img = torch.Tensor(image).unsqueeze(0).permute((0, 3, 1, 2))
    image_tokens = make_patches(img, 14)
    image_tokens = image_tokens.squeeze()

    n_channels = image_tokens.shape[1]  # 3 for this case.

    for k in range(top_k):
        token_index = sorted_indices[k]
        
        # Take each channel and replace with channel mean.
        for ch in range(n_channels):
            image_tokens[token_index][ch].fill_(channel_mean[ch])

    # Reshape back to an image.
    image_tokens = image_tokens.permute((0, 2, 1))
    image_tokens = image_tokens.reshape((196, 16, 16, 3))
    image_tokens = image_tokens.reshape((14, 14, 16, 16, 3))
    image_tokens = image_tokens.permute((0, 2, 1, 3, 4))
    image_tokens = image_tokens.reshape((14 * 16, 14 * 16, 3))
    
    return image_tokens