import torch
import torch.nn as nn

from xai.lrp_rules import output_relevance
from xai.lrp_utils import nan2zero, apply_eps


class xMIL(nn.Module):
    """
    the base class for explanation classes.
    """

    def __init__(self):
        super().__init__()

    def set_explained_class(self, batch):
        return (
            batch["targets"].item()
            if self.explained_class is None
            else self.explained_class
        )

    @staticmethod
    def lrp_gi(
        activations,
        explained_class,
        contrastive_class,
        explained_rel="contrastive",
        eps=1.0e-5,
        verbose=True,
    ):
        logits = activations["out"]["input"].clone()
        relevance_out = output_relevance(
            logits,
            explained_rel=explained_rel,
            explained_class=explained_class,
            contrastive_class=contrastive_class,
            verbose=verbose,
        )
        relevance, R = explain_lrp_gi(
            activations, relevance_out, eps=eps, verbose=verbose
        )
        bag_relevance = relevance.sum(-1).detach().cpu().numpy()
        return bag_relevance, R

    @staticmethod
    def gradient_x_input(features, logit):
        logit.sum().backward()
        explanations = features * features.grad
        return explanations.sum(-1).detach().cpu().numpy()

    @staticmethod
    def squared_grad(features, logit):
        logit.sum().backward()
        explanations = features.grad**2
        return explanations.sum(-1).detach().cpu().numpy()

    def integrated_gradients(self, ig, features, set_explained_class):
        explanations = ig.attribute(
            features, target=set_explained_class, internal_batch_size=len(features)
        )
        return explanations.sum(-1).detach().cpu().numpy()

    @staticmethod
    def perturbation_scores(
        batch, perturbation_method, forward_fn, explained_class, explained_rel="softmax"
    ):
        num_batches, num_patches = len(batch["bag_size"]), batch["bag_size"][0]
        assert num_batches == 1
        scores = []
        for patch_idx in range(num_patches):
            if perturbation_method == "keep":
                keep_idx = [patch_idx]
                bag_sizes = torch.tensor([1])
            elif perturbation_method == "drop":
                keep_idx = list(range(patch_idx)) + list(
                    range(patch_idx + 1, num_patches)
                )
                bag_sizes = torch.tensor([num_patches - 1])
            else:
                raise ValueError(f"Unknown perturbation method: {perturbation_method}")
            features = batch["features"][..., keep_idx, :]
            preds = forward_fn(features, bag_sizes).detach().cpu()
            if explained_rel == "softmax":
                preds = torch.softmax(preds, dim=-1)
            scores.append(preds[:, explained_class])
        scores = torch.cat(scores, dim=0)
        if perturbation_method == "drop":
            features, bag_sizes = batch["features"], batch["bag_size"]
            preds = forward_fn(features, bag_sizes).detach().cpu()
            if explained_rel == "softmax":
                preds = torch.softmax(preds, dim=-1)
            scores = preds[0, explained_class] - scores
        return scores.numpy()

    @staticmethod
    def random_scores(batch):
        return torch.normal(
            mean=torch.zeros(batch["bag_size"][0]), std=torch.ones(batch["bag_size"][0])
        ).numpy()

    def explain_lrp(self, batch, verbose):
        raise NotImplementedError()

    def attention_map(self, batch):
        raise NotImplementedError()

    def explain_gi(self, batch):
        raise NotImplementedError()

    def explain_integrated_gradients(self, batch):
        raise NotImplementedError()

    def explain_squared_grad(self, batch):
        raise NotImplementedError()

    def explain_perturbation(self, batch, perturbation_method):
        raise NotImplementedError()

    def explain_patch_scores(self, batch):
        raise NotImplementedError()

    def get_heatmap(self, batch, heatmap_type, verbose=False):
        if heatmap_type == "attention" or heatmap_type == "attention_rollout":
            patch_scores = self.attention_map(batch)
        elif heatmap_type == "lrp":
            patch_scores, _, _ = self.explain_lrp(batch, verbose=verbose)
        elif heatmap_type == "gi":
            patch_scores = self.explain_gi(batch)
        elif heatmap_type == "grad2":
            patch_scores = self.explain_squared_grad(batch)
        elif heatmap_type == "ig":
            patch_scores = self.explain_integrated_gradients(batch)
        elif heatmap_type == "perturbation_keep" or heatmap_type == "occlusion_keep":
            patch_scores = self.explain_perturbation(batch, "keep")
        elif heatmap_type == "perturbation_drop":
            patch_scores = self.explain_perturbation(batch, "drop")
        elif heatmap_type == "patch_scores":
            patch_scores = self.explain_patch_scores(batch)
        elif heatmap_type == "random":
            patch_scores = xMIL.random_scores(batch)
        else:
            raise ValueError(
                f"Heatmap type not supported for attention mil model: {heatmap_type}"
            )

        return patch_scores

    def get_heatmap_zero_centered(self, heatmap_type):
        """
        helper for retrieving whether the heatmap needs zero centring
        """
        if heatmap_type == "attention" or heatmap_type == "attention_rollout":
            zero_centered = False
        elif heatmap_type == "lrp":
            zero_centered = True
        elif heatmap_type == "gi":
            zero_centered = True
        elif heatmap_type == "grad2":
            zero_centered = False
        elif heatmap_type == "perturbation_keep" or heatmap_type == "occlusion_keep":
            zero_centered = False
        elif heatmap_type == "perturbation_drop":
            zero_centered = False
        else:
            raise ValueError(
                f"Heatmap type not supported for the model: {heatmap_type}"
            )
        return zero_centered


def explain_lrp_gi(activations, relevance_out, eps=1e-5, verbose=True):
    """
    function that takes the activations dictionary and backpropagates relevance_out layer by layer
    """
    R = {"out": relevance_out}
    layer_names = list(reversed(activations.keys()))
    relevance = relevance_out

    if verbose:
        print("propagating relevance back from the following layers ....")
    for i_layer, layer_name in enumerate(layer_names[1:]):
        if verbose:
            print(f"* layer {layer_name}")
        prev_layer_name = layer_names[i_layer]
        act_output = activations[prev_layer_name]

        if "input-p" in act_output and act_output["input-p"] is not None:
            out_y = act_output["input-p"]
        else:
            out_y = act_output["input"]
        y_rel = out_y * (relevance / (out_y + apply_eps(out_y, eps))).data
        y_rel.sum().backward()

        act_input = activations[layer_name]
        act_input_grad = nan2zero(act_input["input-data"].grad)
        relevance = act_input_grad * act_input["input"]
        R[layer_name] = relevance

    return relevance, R
