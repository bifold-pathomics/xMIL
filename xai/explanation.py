import torch
import torch.nn as nn
import numpy as np

from xai.lrp_rules import output_relevance
from xai.lrp_utils import nan2zero, apply_eps


class xMIL(nn.Module):
    """
    the base class for explanation classes. It includes:
    * explain_gi: the method takes the activations dictionary and backpropagates relevance using gradient x input rule.
    """

    def __init__(self, head_type):
        super().__init__()
        self.head_type = head_type

    def set_explained_class(self, batch):
        if self.head_type == "classification":
            return (
                batch["targets"].item()
                if self.explained_class is None
                else self.explained_class
            )
        elif self.head_type in ["regression", "survival"]:
            return 0

    @staticmethod
    def lrp_gi(
        activations,
        explained_class,
        contrastive_class,
        explained_rel="logit",
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
        explanations_vector = features * features.grad
        explanations_bag = explanations_vector.sum(-1).detach().cpu().numpy()
        return (
            explanations_bag.squeeze(),
            explanations_vector.detach().cpu().numpy().squeeze(),
        )

    @staticmethod
    def squared_grad(features, logit):
        logit.sum().backward()
        explanations_vector = features.grad**2
        explanations_bag = explanations_vector.sum(-1).detach().cpu().numpy()
        return (
            explanations_bag.squeeze(),
            explanations_vector.detach().cpu().numpy().squeeze(),
        )

    def integrated_gradients(self, ig, features, set_explained_class):
        explanations_vector = ig.attribute(
            features, target=set_explained_class, internal_batch_size=len(features)
        )
        explanations_bag = explanations_vector.sum(-1).detach().cpu().numpy()
        return (
            explanations_bag.squeeze(),
            explanations_vector.detach().cpu().numpy().squeeze(),
        )

    def perturbation_scores(
        self, batch, perturbation_method, explained_class, explained_rel="softmax"
    ):
        num_batches, num_patches = len(batch["bag_size"]), batch["bag_size"][0]
        assert num_batches == 1
        scores = []
        for patch_idx in range(num_patches):
            if perturbation_method == "keep":
                keep_idx = [patch_idx]
                bag_sizes = torch.tensor([1]).to(self.device)
            elif perturbation_method == "drop":
                keep_idx = list(range(patch_idx)) + list(
                    range(patch_idx + 1, num_patches)
                )
                bag_sizes = torch.tensor([num_patches - 1]).to(self.device)
            else:
                raise ValueError(f"Unknown perturbation method: {perturbation_method}")
            features = batch["features"][..., keep_idx, :]
            features, bag_sizes = features.to(self.model.device), bag_sizes.to(
                self.model.device
            )
            if explained_rel == "survival":
                _, _, risk_scores = self.model.forward_fn(features, bag_sizes)
                scores.append(risk_scores.detach().cpu())
            else:
                preds = self.model.forward_fn(features, bag_sizes).detach().cpu()
                if explained_rel == "softmax":
                    preds = torch.softmax(preds, dim=-1)
                scores.append(preds[:, explained_class])
        scores = torch.cat(scores, dim=0)
        if perturbation_method == "drop":
            features, bag_sizes = batch["features"].to(self.model.device), batch[
                "bag_size"
            ].to(self.model.device)
            if explained_rel == "survival":
                _, _, risk_scores = self.model.forward_fn(features, bag_sizes)
                scores = risk_scores.detach().cpu() - scores
            else:
                preds = self.model.forward_fn(features, bag_sizes).detach().cpu()
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

    def explain_squared_grad(self, batch):
        raise NotImplementedError()

    def explain_integrated_gradients(self, batch):
        raise NotImplementedError()

    def explain_perturbation(self, batch, perturbation_method):
        self.model.eval()
        explained_class = self.set_explained_class(batch)
        return self.perturbation_scores(
            batch, perturbation_method, explained_class, self.explained_rel
        )

    def explain_patch_scores(self, batch):
        raise NotImplementedError()

    def get_heatmap(self, batch, heatmap_type, verbose=False):
        if heatmap_type == "attention" or heatmap_type == "attention_rollout":
            patch_scores = self.attention_map(batch)
            patch_scores_vector = None
        elif heatmap_type == "lrp":
            patch_scores, R, _ = self.explain_lrp(batch, verbose=verbose)
            patch_scores_vector = list(R.values())[-1].squeeze().detach().cpu().numpy()
        elif heatmap_type == "gi":
            patch_scores, patch_scores_vector = self.explain_gi(batch)
        elif heatmap_type == "grad2":
            patch_scores, patch_scores_vector = self.explain_squared_grad(batch)
        elif heatmap_type == "ig":
            patch_scores, patch_scores_vector = self.explain_integrated_gradients(batch)
        elif heatmap_type == "perturbation_keep" or heatmap_type == "occlusion_keep":
            patch_scores = self.explain_perturbation(batch, "keep")
            patch_scores_vector = None
        elif heatmap_type == "perturbation_drop":
            patch_scores = self.explain_perturbation(batch, "drop")
            patch_scores_vector = None
        elif heatmap_type == "patch_scores":
            patch_scores = self.explain_patch_scores(batch)
            patch_scores_vector = None
        elif heatmap_type == "random":
            patch_scores = xMIL.random_scores(batch)
            patch_scores_vector = None
        else:
            raise ValueError(
                f"Heatmap type not supported for attention mil model: {heatmap_type}"
            )

        if patch_scores_vector is None:
            patch_scores_vector = np.full(patch_scores.shape, np.nan)
        return patch_scores, patch_scores_vector

    def get_heatmap_zero_centered(self, heatmap_type):
        if heatmap_type == "attention" or heatmap_type == "attention_rollout":
            zero_centered = False
        elif heatmap_type == "lrp":
            zero_centered = True
        elif heatmap_type == "gi":
            zero_centered = True
        elif heatmap_type == "grad2":
            zero_centered = False
        elif heatmap_type == "ig":
            zero_centered = True
        elif heatmap_type == "perturbation_keep" or heatmap_type == "occlusion_keep":
            zero_centered = True
        elif heatmap_type == "perturbation_drop":
            zero_centered = True
        else:
            raise ValueError(
                f"Heatmap type not supported for the model: {heatmap_type}"
            )
        return zero_centered


def explain_lrp_gi(activations, relevance_out, eps=1e-5, verbose=True):
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


class RegressionModelRestructuring(nn.Module):
    # ToDo: this class is not used in other parts. but it is tested and works.
    def __init__(
        self,
        model,
        method="flood",
        step_width=0.005,
        max_iter=10e4,
        normalize_top=False,
        device="cpu",
    ):
        super().__init__()
        self.model = model
        self.method = method
        self.step_width = step_width
        self.max_iter = max_iter
        self.normalize_top = normalize_top
        self.device = device

    def prepare_model_for_regression(self, input_, bag_size, y_ref):
        a_ref = self.find_a_ref(input_, bag_size, y_ref=y_ref)
        return self.restructure_model(a_ref)

    def restructure_model(self, a_ref):
        """
        Restructures the incoming and outgoing layers of a PyTorch model according to [1].

        Parameters:
            model (torch.nn.Sequential): Model to be restructured. Assumption for top layer structure [Linear(i,j), ReLU, Linear(j,1)]
            a_ref (torch.Tensor): Reference offset used for restructuring. Assumption a_ref.shape = (1, j)
            in_layer (int): The index of the incoming layer to be restructured. Default is -3.
            out_layer (int): The index of the outgoing layer to be restructured. Default is -1.

        Returns:
            nn.Module: A restructured version of the input model.
        (c) https://github.com/sltzgs/xai-regression
        """

        # get weights and biases

        linear_layer_in = self.model.get_out_layers()[-1][-2]
        W_in = linear_layer_in.weight
        W_out = self.model.get_linear_head().weight
        bias_in = linear_layer_in.bias.view(1, -1)

        # manipulate biases in incoming layer
        # a_ref multiplied by -1, 0, and 1 before being added to biases
        # biases multiplied with incoming weight manipulation (1, -1, -1)
        bias_in = torch.cat((bias_in - a_ref, -bias_in, -bias_in + a_ref), 1).to(
            self.device
        )

        # initialization of new top layer structure
        top_in = nn.Linear(W_in.shape[1], W_in.shape[0] * 3)
        top_act = nn.ReLU()
        top_out = nn.Linear(W_out.shape[1] * 3, W_out.shape[0])

        with torch.no_grad():
            top_in.weight = nn.Parameter(
                torch.cat((W_in, -W_in, -W_in), 0).to(self.device)
            )  # incoming weights multiplied by 1, -1, and -1
            top_in.bias = nn.Parameter(bias_in)
            top_out.weight = nn.Parameter(
                torch.cat((W_out, W_out, -W_out), 1).to(self.device)
            )  # outgoing weights multiplied by 1, 1, and -1
            top_out.bias = nn.Parameter(torch.zeros([1, 1]).to(self.device))

        new_model = self.model_without_head()
        out_layer_old = self.model.get_out_layers()
        out_layer_new = [out_layer_old[i] for i in range(len(out_layer_old) - 1)] + [
            nn.Sequential(top_in, nn.ReLU())
        ]
        new_model.set_out_layers(nn.Sequential(*out_layer_new))
        new_model.set_linear_head(top_out)

        # new_model = nn.Sequential(*(list(self.model[:in_layer]) + [top_in, top_act, top_out] + model_top_left))

        return new_model

    def find_a_ref(self, input_, bag_size, y_ref=0, verbose=False):
        """
        find_a_ref : function to find the reference offset 'a_ref' for a given model, input and reference value 'y_ref'

        Parameters:
        model (torch.nn.Sequential): PyTorch model to be used for the computation
        input_ (torch.Tensor): Input sample for which we search the offset 'a_ref'
        y_ref (float, optional): Reference value for the output y (default: 0)
        method (str, optional): Method used to find the reference value for the output a (default: 'flood')
        step_width (float, optional): Step width used in the 'flood' method (default: 0.005)
        max_it (int, optional): Maximum number of iterations before stopping the loop and displaying a warning message. (default: 10e4)
        normalize_top (bool, optional): If True, the model top layers are rescaled to +1/-1 as weights in the last layer. While this is not
                                        described in the paper, we found that it is closer to our intuition of the flooding rule. (default: False)

        Returns:
        torch.Tensor : Found reference offset 'a_ref'
        (c) https://github.com/sltzgs/xai-regression
        """
        self.model.eval()

        print(
            f"y = {np.round(self.model.forward_fn(input_, bag_size).detach().cpu().numpy(), 2)},  y_ref = {y_ref}"
        )

        # if normalize_top == True:
        #     model = rescale_top(model)

        if self.method == "flood":  # other methods for finding a_ref can be added

            y_t_ = self.model.forward_fn(input_, bag_size)
            model_c = self.model_without_head()
            a_ref_ = model_c.forward_fn(input_, bag_size)

            update = (torch.ones(a_ref_.shape[1])).to(self.device) * self.step_width

            counter = 0

            if y_t_ > y_ref:

                while y_t_ > y_ref:

                    a_ref_ = torch.max(
                        torch.zeros(a_ref_.shape[1]).to(self.device), (a_ref_ - update)
                    )

                    y_t_ = self.model.get_linear_head().forward(a_ref_)  # Achtung ***
                    counter += 1
                    if verbose:
                        print(f"iteration {counter} - y_t: {y_t_}", end="\r")
                    if counter > self.max_iter:
                        print(
                            f"! reference value {y_ref} was not reached within {round(self.max_iter)} iterations!"
                        )
                        break

            else:

                while y_t_ < y_ref:

                    a_ref_ = torch.max(
                        torch.zeros(a_ref_.shape[1]).to(self.device), (a_ref_ + update)
                    )

                    y_t_ = self.model.get_linear_head().forward(a_ref_)  # Achtung ***
                    counter += 1
                    if verbose:
                        print(f"iteration {counter} - y_t: {y_t_}", end="\r")
                    if counter > self.max_iter:
                        print(
                            f"! reference value {y_ref} was not reached within {round(self.max_iter)} iterations!"
                        )
                        break

        else:
            NotImplementedError()

        return a_ref_

    def model_without_head(self):
        buffer = io.BytesIO()
        torch.save(self.model, buffer)
        buffer.seek(0)
        model_c = torch.load(buffer)
        model_c.set_linear_head(None)
        return model_c
