import copy

import torch
import torch.nn.functional as F

from xai.lrp_utils import apply_eps


def gamma_layer(x, layer, gamma=0, eps=1e-5, no_bias=True):
    """
    helper function for explanation methods.
    the output is equal to layer(x), the difference is in the derivation of the output with reqards to x.
    Important for gradient x input when applying it to a cascade of blocks.
    """
    z = layer(x)
    zp = modified_linear_layer(layer, gamma, no_bias=no_bias)(x)
    return zp * (z / (zp + apply_eps(zp, eps))).data


def modified_linear_layer(layer, gamma, no_bias=True):
    """
    Based on: https://git.tu-berlin.de/gmontavon/lrp-tutorial/-/tree/main

    :param layer: (nn.Linear)
    :param gamma: (float)
    :param no_bias: (bool)
    :return: (nn.Linear)
    """
    rho = lambda p: p + gamma * p.clamp(min=0)
    layer_new = copy.deepcopy(layer)
    layer_new.weight = torch.nn.Parameter(rho(layer.weight))
    layer_new.bias = (
        None
        if layer.bias is None
        else torch.nn.Parameter(layer.bias * 0 if no_bias else rho(layer.bias))
    )
    return layer_new


def check_relevance_conservation(rel_dict, verbose=True):
    if verbose:
        print("sum of relevance values at layers:")
    rel_sum = dict()
    for layer, relevance in rel_dict.items():
        rel_sum[layer] = relevance.sum().item()
        if verbose:
            print(f"* {layer}: {rel_sum[layer]}")
    return rel_sum


def output_relevance(
    logits,
    explained_rel="logit",
    explained_class=1,
    contrastive_class=None,
    verbose=False,
):
    """
    function for generating the output relevance. It can be the logit of the class or the contrastive relevance.
    [citation] contrastive rule implemented based on equations on page 202 of Montavon et al 2019.

    :param logits: logits of the classification head
    :param explained_rel: can be "contrastive" or "logit"
    :param explained_class: the class to compute the relevance values for.
    :param contrastive_class: in 'contrastive' relevance, explained_class is contrasted to contrastive_class
    :param verbose: bool
    :return:
    """
    if not isinstance(explained_class, int):
        return ValueError("the explained class should be an integer.")

    n_classes = logits.shape[1]
    not_explained_classes = [i for i in range(n_classes) if i != explained_class]
    if explained_rel == "contrastive":
        if contrastive_class is None and len(not_explained_classes) == 1:
            contrastive_class = not_explained_classes[0]
            if verbose:
                print(
                    f"the explained class is {explained_class}, but no contrastive class determined."
                    f"the other class is taken as the contrastive class."
                )
        elif contrastive_class is None:
            return ValueError(
                f"for contrastive explanation, at least one class should be given as "
                f"the contrastive class."
            )

        if verbose:
            print(
                f"the explained class is {explained_class}, and the contrastive class is {contrastive_class}"
            )

        c_cp_onehot = F.one_hot(
            torch.tensor([explained_class]), num_classes=n_classes
        ).to(logits.device) + F.one_hot(
            torch.tensor([contrastive_class]), num_classes=n_classes
        ).to(
            logits.device
        )

        relevance_out = c_cp_onehot * logits
        relevance_out[:, contrastive_class] *= -1
        z_c_all = logits[:, explained_class] - logits[:, not_explained_classes]
        relevance_out = (
            relevance_out
            * torch.exp(-relevance_out).sum(-1)
            / torch.exp(-z_c_all).sum(-1)
        )

    elif explained_rel == "logit":
        relevance_out = logits
        relevance_out[:, not_explained_classes] = 0
    else:
        raise ValueError(
            f"{explained_rel} is not implemented as a relevance computation method."
        )

    return relevance_out
