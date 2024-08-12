import torch
import numpy as np


def beyond_intuition_tokenwise(model, x, device, index=None, steps=20, start_layer=6, samples=20, noise=0.2, mae=False, ssl=False, dino=False):
    x = x.to(device)

    b = x.shape[0]
    output = model(x, register_hook=True)
    if index is None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
    one_hot[np.arange(b), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
    one_hot = torch.sum(one_hot * output)

    model.zero_grad()
    one_hot.backward(retain_graph=True)

    _, num_head, num_tokens, _ = model.blocks[-1].attn.get_attention_map().shape

    R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).to(device)
    for nb, blk in enumerate(model.blocks):
        if nb < start_layer - 1:
            continue
        z = blk.get_input()
        vproj = blk.attn.vproj
        order = torch.linalg.norm(vproj, dim=-1).squeeze() / torch.linalg.norm(z, dim=-1).squeeze()
        m = torch.diag_embed(order)
        cam = blk.attn.get_attention_map()
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(0)
        R = R + torch.matmul(torch.matmul(cam.to(device), m.to(device)), R.to(device))

    if ssl:
        if mae:
            return R[:, 1:, 1:].abs().mean(axis=1)
        elif dino:
            return (R[:, 1:, 1:].abs().mean(axis=1) + R[:, 0, 1:].abs())
        else:
            return R[:, 0, 1:].abs()

    total_gradients = torch.zeros(b, num_head, num_tokens, num_tokens, device=device)
    for alpha in np.linspace(0, 1, steps):
        data_scaled = x * alpha
        output = model(data_scaled, register_hook=True)
        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
        one_hot = torch.sum(one_hot * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)

        gradients = model.blocks[-1].attn.get_attn_gradients()
        total_gradients += gradients

    W_state = (total_gradients / steps).clamp(min=0).mean(1).reshape(b, num_tokens, num_tokens)
    R = W_state * R.abs()

    if mae:
        return R[:, 1:, 1:].mean(axis=1)
    elif dino:
        return (R[:, 1:, 1:].mean(axis=1) + R[:, 0, 1:])
    else:
        return R[:, 0, 1:]