import torch
import numpy as np

def beyond_intuition_headwise(model, x, device, target_cls_idx=None, steps=20, start_layer=4, samples=20, noise=0.2, mae=False, dino=False, ssl=False):
    x = x.to(device)
    output = model(x, register_hook=True)
    if target_cls_idx is None:
        target_cls_idx = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((x.shape[0], output.size()[-1]), dtype=np.float32)
    one_hot[np.arange(x.shape[0]), target_cls_idx] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
    one_hot = torch.sum(one_hot * output)

    model.zero_grad()
    one_hot.backward(retain_graph=True)

    _, num_head, num_tokens, _ = model.blocks[-1].attn.get_attention_map().shape
    R = torch.eye(num_tokens, num_tokens).expand(x.shape[0], num_tokens, num_tokens).to(device)
    for nb, blk in enumerate(model.blocks):
        if nb < start_layer - 1:
            continue
        grad = blk.attn.get_attn_gradients().to(device)  # Gradient only required to calculate Ih. Ih is the importance factor for an attention map. As each head have a single attn map, Ih works as the important factor for a head.
        grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
        cam = blk.attn.get_attention_map().to(device)
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])


        # ⚠️⚠️⚠️⚠️ With the help of a method proposed in ScoreCAM paper to evaluate importance score (mentioned in Fig-3's title prolly), how valid Ih is.
        Ih = torch.mean(torch.matmul(cam.transpose(-1, -2), grad).abs(), dim=(-1, -2))
        Ih = Ih / torch.sum(Ih)
        cam = torch.matmul(Ih, cam.reshape(num_head, -1)).reshape(num_tokens, num_tokens)

        R = R + torch.matmul(cam.to(device), R.to(device))

    if ssl:
        if mae:
            return R[:, 1:, 1:].abs().mean(axis=1)
        elif dino:
            return (R[:, 1:, 1:].abs().mean(axis=1) + R[:, 0, 1:].abs())
        else:
            return R[:, 0, 1:].abs()

    # Reasoning Feedback
    total_gradients = torch.zeros(x.shape[0], num_head, num_tokens, num_tokens).to(device)
    for alpha in np.linspace(0, 1, steps):
        data_scaled = x * alpha

        output = model(data_scaled, register_hook=True)
        one_hot = np.zeros((x.shape[0], output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(x.shape[0]), target_cls_idx] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
        one_hot = torch.sum(one_hot * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)

        gradients = model.blocks[-1].attn.get_attn_gradients().to(device)
        total_gradients += gradients

    W_state = (total_gradients / steps).clamp(min=0).mean(1).reshape(x.shape[0], num_tokens, num_tokens)
    R = W_state * R

    if mae:
        return R[:, 1:, 1:].mean(axis=1)
    elif dino:
        return (R[:, 1:, 1:].mean(axis=1) + R[:, 0, 1:])
    else:
        return R[:, 0, 1:]
