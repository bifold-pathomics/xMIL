import torch
import torch.nn as nn


def set_lrp_params(lrp_params):
    """
    helper function to set the dictionary containing LRP parameters
    Args:
        lrp_params: dictionary  must include keys 'gamma', 'eps', 'no_bias'

    Returns: lrp_params with missing keys filled

    """
    if lrp_params is None:
        lrp_params = {"gamma": 0, "eps": 1e-8, "no_bias": True}
    else:
        if "gamma" not in lrp_params:
            lrp_params["gamma"] = 0
        if "eps" not in lrp_params:
            lrp_params["eps"] = 1e-8
        if "no_bias" not in lrp_params:
            lrp_params["no_bias"] = True
        else:
            lrp_params["no_bias"] = bool(lrp_params["no_bias"])
    return lrp_params


def set_detach_norm(detach_norm):
    """
    helper function to set the dictionary containing the keys to determine the detachments in layer norm
    Args:
        detach_norm: dictionary, must include keys 'mean' and 'std'

    Returns: detach_norm with missing keys filled

    """
    if detach_norm is None:
        detach_norm = {"mean": False, "std": True}
    else:
        if "mean" not in detach_norm:
            detach_norm["mean"] = False
        if "std" not in detach_norm:
            detach_norm["std"] = True
    return detach_norm


def var_data_requires_grad(x):
    """
    helper function for explanation methods. detaches the input variable from computational graph
    and then enables its grad computation again

    """
    xdata = x.data
    xdata.requires_grad_(True)
    return xdata


def nan2zero(input_tensor):
    input_clone = input_tensor.clone()
    input_clone[torch.isnan(input_tensor)] = 0
    return input_clone


def apply_eps(x, eps):
    return torch.where(x >= 0, 1, -1) * eps


def layer_norm(detach_norm=None, dim=None, weight=1, bias=0, verbose=False):
    if detach_norm is None or (not detach_norm["mean"] and not detach_norm["std"]):
        return nn.LayerNorm(dim)
    if verbose:
        print(f"A detachment is done at the layer norm: {detach_norm}!")
    return LayerNormDetach(detach_norm=detach_norm, weight=weight, bias=bias)


class LayerNormDetach(nn.Module):
    """
    (c) modified from https://github.com/AmeenAli/XAI_Transformers/blob/main/utils.py
    """

    def __init__(self, detach_norm, weight=1, bias=0, eps=1e-5):
        super().__init__()
        self.detach_norm = detach_norm
        self.eps = eps
        self.weight = weight
        self.bias = bias

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        if self.detach_norm["mean"]:
            x_mean = x_mean.detach()
        if self.detach_norm["std"]:
            x_std = x_std.detach()
        return (x - x_mean) / (
            x_std + apply_eps(x_std, self.eps)
        ) * self.weight + self.bias
