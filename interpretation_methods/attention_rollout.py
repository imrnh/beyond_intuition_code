import torch


"""
    adding residual consideration- code adapted from https://github.com/samiraabnar/attention_flow
"""
def compute_rollout_attention(all_layer_matrices, start_layer=0):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    device = all_layer_matrices[0].device
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(device)
    all_layer_matrices = [all_layer_matrices[i] + eye for i in range(len(all_layer_matrices))]
    matrices_aug = [all_layer_matrices[i] / all_layer_matrices[i].sum(dim=-1, keepdim=True)
                    for i in range(len(all_layer_matrices))]
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, len(matrices_aug)):
        joint_attention = matrices_aug[i].bmm(joint_attention)
    return joint_attention


def rollout_attention(self, x, device, start_layer=0, mae=False):
    x = x.to(device)
    self.model(x)
    blocks = self.model.blocks
    all_layer_attentions = []
    for blk in blocks:
        attn_heads = blk.attn.get_attention_map().to(device)
        avg_heads = (attn_heads.sum(dim=1) / attn_heads.shape[1]).detach()
        all_layer_attentions.append(avg_heads)
    rollout = compute_rollout_attention(all_layer_attentions, start_layer=start_layer)
    if mae:
        return rollout[:, 1:, 1:].mean(1)
    else:
        return rollout[:, 0, 1:]