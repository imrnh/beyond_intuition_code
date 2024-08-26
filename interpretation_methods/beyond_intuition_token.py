import torch
import numpy as np


def beyond_intuition_tokenwise(model, x, device, onehot, index=None, steps=20, start_layer=6, samples=20, noise=0.2, mae=False, ssl=False, dino=False, taken_head_idx = None):

    # A dictionary to track down all the output at every step, so we can visualize it.
    stat_dict = {
        "input": x,
        "steps": steps,
        "start_layer": start_layer,
    }

    x = x.to(device)
    b = x.shape[0]  # Batch size

    _, num_head, num_tokens, _ = model.blocks[-1].attn.get_attention_map().shape
    R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).to(device)
    
    stat_dict['final_blk_attn_shape'] = model.blocks[-1].attn.get_attention_map().shape
    stat_dict['pre_attn_perception_R'] = R
    stat_dict['attention_perception'] = dict()


    for blk_idx, blk in enumerate(model.blocks):
        if blk_idx < start_layer - 1:
            continue

        # Calculate alpha in the paper.
        z = blk.get_input()  # We can even call it z.
        vproj = blk.attn.vproj  # vproj = Z * Wv * W(l).
        order = torch.linalg.norm(vproj, dim=-1).squeeze() / torch.linalg.norm(z, dim=-1).squeeze()  # Z * Wv * W(l) / Z    --->  V / Z.  # Shape --> (197,)
        m = torch.diag_embed(order)  # Converted the order into a diagonal matrix. # Shape (197, 197)

        # Get attention map A_i which will be multiplied by alpha_i.
        cam = blk.attn.get_attention_map()  # Shape --> (batch_size, num_heads, num_tokens, num_tokens)   (12, 197, 197)
        cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])  # Make the shape (num_heads, num_tokens, num_tokens).  

        
        if taken_head_idx is None:
            cam = cam.mean(0)  # Then take headwise mean and make it of shape (num_tokens, num_tokens).
            # That is, take average of all head's attention for a token.
        else:
            cam = cam[taken_head_idx]  # Select specific head only.

        # Also, try taking variance of gradient of attention and multiply then with that.
        O_t = torch.matmul(cam.to(device), m.to(device))  # O = AZW   >>>  cam shape (197, 197) |  m shape (197, 197)
        R = R + torch.matmul(O_t, R.to(device))

        
        stat_dict['attention_perception'][f"rOps_{blk_idx}"] = {
            "z": z,
            "m": m,
            "order": order,
            "vproj": vproj,
            "cam_noreshape": cam,
            "mean_cam": cam.mean(0),  # Headwise mean
            "O": O_t,
            "R": R,
        }


    if ssl:
        if mae:
            return R[:, 1:, 1:].abs().mean(axis=1)
        elif dino:
            return R[:, 1:, 1:].abs().mean(axis=1) + R[:, 0, 1:].abs()
        else:
            return R[:, 0, 1:].abs()

    stat_dict["R_attn_perception"] = R
    stat_dict['reasoning_feedback'] = dict()

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

        stat_dict['reasoning_feedback'][f"alpha_{alpha}"] = {
            "data_scaled": data_scaled,
            "output": output,
            "one_hot": one_hot,
            "gradients": gradients,
            "gradient_shape":  model.blocks[-1].attn.get_attn_gradients().shape,
            "total_gradients": total_gradients,
        }

    W_state = (total_gradients / steps).clamp(min=0).mean(1).reshape(b, num_tokens, num_tokens)
    R = W_state * R.abs()

    stat_dict["W_state"] = W_state
    stat_dict["R_resoaning_feedback"] = R


    if mae:
        R_MAE = R[:, 1:, 1:].mean(axis=1)
        stat_dict["R"] = (R_MAE, "MAE")
        return R_MAE, stat_dict

    elif dino:
        R_DINO = (R[:, 1:, 1:].mean(axis=1) + R[:, 0, 1:])
        stat_dict["R"] = (R_DINO, "DINO")
        return R_DINO, stat_dict

    else:
        R_general = R[:, 0, 1:]
        stat_dict["R"] = (R_general, "General Case")
        return R_general, stat_dict
