import torch
import numpy as np


def beyond_intuition_tokenwise(model, x, device, index=None, steps=20, start_layer=6, samples=20, noise=0.2, mae=False, ssl=False, dino=False):

    # A dictionary to track down all the output at every step, so we can visualize it.
    TRACKER_DICTIONARY = {
        "input": x.tolist(),
        "steps": steps,
        "start_layer": start_layer,
    }


    x = x.to(device)

    b = x.shape[0]  # Batch size
    output = model(x, register_hook=True)
    if index is None:
        index = np.argmax(output.cpu().data.numpy(), axis=-1)

    TRACKER_DICTIONARY["output"] = output.tolist()

    one_hot = np.zeros((b, output.size()[-1]), dtype=np.float32)
    one_hot[np.arange(b), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
    one_hot = torch.sum(one_hot * output)

    TRACKER_DICTIONARY['one_hot'] = one_hot.tolist()

    model.zero_grad()
    one_hot.backward(retain_graph=True)

    _, num_head, num_tokens, _ = model.blocks[-1].attn.get_attention_map().shape

    TRACKER_DICTIONARY['final_blk_attn_shape'] = model.blocks[-1].attn.get_attention_map().shape

    R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).to(device)

    TRACKER_DICTIONARY['R_before_block_ops'] = R.tolist()

    TRACKER_DICTIONARY['Attention_Perception'] = dict()


    for blk_idx, blk in enumerate(model.blocks):
        if blk_idx < start_layer - 1:
            continue

        # Calculate alpha in the paper.
        z = blk.get_input()  # We can even call it z.
        vproj = blk.attn.vproj  # vproj = Z * Wv * W(l).
        order = torch.linalg.norm(vproj, dim=-1).squeeze() / torch.linalg.norm(z, dim=-1).squeeze()  # Z * Wv * W(l) / Z    --->  V / Z.
        m = torch.diag_embed(order)  # Converted the order into a diagonal matrix.

        # Get attention map A_i which will be multiplied by alpha_i.
        cam = blk.attn.get_attention_map()  # Shape --> (batch_size, num_tokens, num_tokens)
        mean_cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1]).mean(0)  # Make the shape (1, batch_size, num_tokens, num_tokens).
        # Then take mean for the first axis and making it of shape (batch_size, num_tokens, num_tokens).
        # But from a sample test, I've seen that it doesn't change the cam tensor somehow.

        O_t = torch.matmul(mean_cam.to(device), m.to(device))  # O = AZW
        R = R + torch.matmul(O_t, R.to(device))

        TRACKER_DICTIONARY['Attention_Perception'][f"rOps_{blk_idx}"] = {
            "z": z.tolist(),
            "m": m.tolist(),
            "order": order.tolist(),
            "vproj": vproj.tolist(),
            "cam_noreshape": cam.tolist(),
            "mean_cam": mean_cam.tolist(),
            "O": O_t.tolist(),
            "R": R.tolist(),
        }



    if ssl:
        if mae:
            return R[:, 1:, 1:].abs().mean(axis=1)
        elif dino:
            return R[:, 1:, 1:].abs().mean(axis=1) + R[:, 0, 1:].abs()
        else:
            return R[:, 0, 1:].abs()

    TRACKER_DICTIONARY["AP_R_after_modelwise_transformation"] = R.tolist()

    TRACKER_DICTIONARY['Reasoning_Feeback'] = dict()


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

        TRACKER_DICTIONARY['Reasoning_Feeback'][f"alpha_{alpha}"] = {
            "data_scaled": data_scaled.tolist(),
            "output": output.tolist(),
            "one_hot": one_hot.tolist(),
            "gradients": gradients.tolist(),
            "gradient_shape":  model.blocks[-1].attn.get_attn_gradients().shape,
            "total_gradients": total_gradients.tolist(),
        }

    W_state = (total_gradients / steps).clamp(min=0).mean(1).reshape(b, num_tokens, num_tokens)
    R = W_state * R.abs()

    TRACKER_DICTIONARY["W_state"] = W_state.tolist()
    TRACKER_DICTIONARY["R_after_resfeedback"] = R.tolist()


    if mae:
        R_MAE = R[:, 1:, 1:].mean(axis=1)
        TRACKER_DICTIONARY["R_final"] = (R_MAE.tolist(), "MAE")
        return R_MAE, TRACKER_DICTIONARY

    elif dino:
        R_DINO = (R[:, 1:, 1:].mean(axis=1) + R[:, 0, 1:])
        TRACKER_DICTIONARY["R_final"] = (R_DINO.tolist(), "DINO")
        return R_DINO, TRACKER_DICTIONARY

    else:
        R_general = R[:, 0, 1:]
        TRACKER_DICTIONARY["R_final"] = (R_general.tolist(), "General Case")
        return R_general, TRACKER_DICTIONARY
