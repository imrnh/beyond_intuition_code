import numpy as np
import torch

def integrated_gradient(model, x, device, t_cls_idx=None, steps=20):
    x = x.to(device)

    b = x.shape[0]
    output = model(x, device, register_hook=True)
    if t_cls_idx is None:
        t_cls_idx = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
    one_hot[np.arange(b), t_cls_idx] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
    one_hot = torch.sum(one_hot * output)

    model.zero_grad()
    one_hot.backward(retain_graph=True)

    _, num_head, num_tokens, _ = model.blocks[-1].attn.get_attention_map().shape

    total_gradients = torch.zeros([b, 1, 224, 224], device=device)
    for alpha in np.linspace(0, 1, steps):
        data_scaled = x * alpha
        output = model(data_scaled, register_hook=True)
        one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
        one_hot[np.arange(b), t_cls_idx] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
        one_hot = torch.sum(one_hot * output)

        model.zero_grad()
        one_hot.backward(retain_graph=True)

        gradients = model.get_x_gradients().sum(1).unsqueeze(1)
        total_gradients += gradients

    W_state = total_gradients / steps

    return W_state