"""
    For a given dataset, corrupt each image using saliency map.

"""
import torch
import torch.nn as nn
import numpy as np

def get_all_channels_mean(image):
    channel_mean = []
    permuted_image = torch.permute(image, (2,1,0))
    for ch in range(image.shape[-1]):
        channel_mean.append(permuted_image[ch].mean())

    return channel_mean



def get_corrupted_image(image, saliency, channel_mean, corrupt_percentage):
    saliency_flat = torch.flatten(saliency)
    saliency_sorted, sorted_indices = torch.sort(saliency_flat)
    image_permuted = torch.permute(image, (2, 1, 0)).unsqueeze(0)  # Reshape image into (1, 3, 224, 224)

    top_k = int(corrupt_percentage * len(saliency_sorted))  # number of classes to be sorted.
    channel_mean = get_all_channels_mean(image)

    image = image.detach().cpu().numpy()
    image_tokens = image.reshape(14, 16, 14, 16, 3)
    image_tokens = image_tokens.transpose(0, 2, 1, 3, 4).reshape(14 * 14, 16, 16, 3) # Rearrange the dimensions to get (196, 3, 16, 16)
    
    n_channels = image_tokens.shape[1]  # 3 for this case.

    for k in range(top_k):
        token_index = sorted_indices[k]
        
        # Take each channel and replace with channel mean.
        for ch in n_channels:
            mean_token = np.full((16, 16), channel_mean[ch])
            image_tokens[token_index][ch] = mean_token


    return image