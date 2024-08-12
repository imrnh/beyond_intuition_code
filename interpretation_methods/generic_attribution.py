import torch
import numpy as np


def generic_attribution(model, x, device,  start_layer=1, t_cls_idx=None, mae=False):
    x = x.to(device)

    b = x.shape[0]
    output = model(x, device,  register_hook=True)
    if t_cls_idx is None:
        t_cls_idx = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
    one_hot[np.arange(b), t_cls_idx] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
    one_hot = torch.sum(one_hot * output)

    model.zero_grad()
    one_hot.backward(retain_graph=True)

    _, num_head, num_tokens, _ = model.blocks[-1].attn.get_attention_map().shape

    R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).to(device)
    for nb, blk in enumerate(model.blocks):
        if nb < start_layer - 1:
            continue

        cam = blk.attn.get_attention_map()
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])
        grad = blk.attn.get_attn_gradients()
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = (grad * cam).mean(0).clamp(min=0)
        R = R + torch.matmul(cam, R)

    if mae:
        return R[:, 1:, 1:].mean(axis=1)
    else:
        return R[:, 0, 1:]