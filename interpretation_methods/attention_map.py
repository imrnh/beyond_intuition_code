import torch

"""
    We need to get current attention map of the model for an input.
    Therefore, no backward pass needed. Just for every forward pass, we map out the attention scores.

"""

def raw_attention_map(model, x, device, mae=False, dino=False):
    x = x.to(device)

    model(x, register_hook=True)  # Forward pass to calculate attention for the given image.
    cam = model.blocks[-1].attn.get_attention_map().to(device)

    if mae:
        cam = cam[0, :, 1:, 1:]
        cam = cam.mean(0).mean(1).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        cam = cam.mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam