import torch
import numpy as np


def cam_attn(model, x, device, target_cls_idx=None, mae=False):
    x = x.to(device)
    output = model(x, register_hook=True)

    if target_cls_idx is None:
        target_cls_idx = np.argmax(output.cpu().data.numpy())

    # Converting output to one_hot vector. That is, only target neuron's activation is kept and other activation eliminated.
    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0][target_cls_idx] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
    one_hot = torch.sum(one_hot * output)

    model.zero_grad()  # Removing currently store gradients as we would recaclualte wrt onehot.
    one_hot.backward(retain_graph=True)

    grad = model.blocks[-1].attn.get_attn_gradients().to(device)
    cam = model.blocks[-1].attn.get_attention_map().to(device)

    if mae:
        cam = cam[0, :, 1:, 1:]
        grad = grad[0, :, 1:, 1:]
        grad = grad.mean(dim=[0, 2], keepdim=True)
        cam = (cam * grad).mean(0).mean(1).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = cam[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad[0, :, 0, 1:].reshape(-1, 14, 14)
        grad = grad.mean(dim=[1, 2], keepdim=True)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

    return cam
