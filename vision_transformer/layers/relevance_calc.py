import torch
import torch.nn as nn


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)  # Making sure no value is zero. min=1e-9 clips anything below 1e-9. max=1e-9 makes sure all the negative gets added Hence only few point close to zero gets eliminated.
    den = den + den.eq(0).type(den.type()) * 1e-9  # .eq(a) means if any element of the tensor equals to a. If any point comes true i.e. any elem equal to zero, the True would be multiplied by 1e-9 and hence 1e-9 for that element would be added
    return a / den * b.ne(0).type(b.type())  # ne = Not equal function


def forward_hook(curr_layer, _input, _output):
    if type(_input[0]) in (list, tuple):
        curr_layer.X = []
        for i in _input[0]:
            x = i.detach()
            x.requires_grad = True
            curr_layer.X.append(x)
    else:
        curr_layer.X = _input[0].detach()
        curr_layer.X.requires_grad = True

    curr_layer.Y = _output


def backward_hook(curr_layer, grad_input, grad_output):
    curr_layer.grad_input = grad_input
    curr_layer.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):  # This method only needed for: Conv2d, Add, IndexSelect, Clone, Cat, RelPropSimple
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, rel, alpha):
        return rel


class RelPropSimple(RelProp):
    def relprop(self, relevance, alpha):
        Z = self.forward(self.X)
        S = safe_divide(relevance, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs


class AddEye(RelPropSimple):
    def forward(self, input): # input of shape B, C, seq_len, seq_len
        return input + torch.eye(input.shape[2]).expand_as(input).to(input.device)
