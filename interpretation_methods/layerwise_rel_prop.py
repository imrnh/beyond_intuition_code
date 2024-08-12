import torch
import numpy as np

"""
    @param x: Input image
    @param target_cls_idx: Index of the class for which the heatmap should be created. If not give, predict the class.

"""
def layerwise_relevance_propagation(model, x, device, target_cls_idx=None, method="transformer_attribution", is_ablation=False, start_layer=0):
    x = x.to(device)
    output = model(x)
    kwargs = {"alpha": 1}

    if target_cls_idx is None:
        target_cls_idx = np.argmax(output.cpu().data.numpy(), axis=-1)

    one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
    one_hot[0, target_cls_idx] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
    one_hot = torch.sum(one_hot * output)

    model.zero_grad()
    one_hot.backward(retain_graph=True)  # Calulate gradient with respect to this given class index.

    one_hot_tensor = torch.tensor(one_hot.detach().cpu().numpy()).to(device)
    relevance = model.relprop(one_hot_tensor, method=method, is_ablation=is_ablation, start_layer=start_layer, **kwargs)

    return relevance
