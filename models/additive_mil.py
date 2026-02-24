"""
(c) Adaptation of the paper: https://arxiv.org/abs/2206.01794.
most parts of the code (except class xAdditiveMIL) taken from supplementary materials from
https://openreview.net/forum?id=5dHQyEcYDgA.
"""

from typing import Sequence

import torch
import torch.nn as nn

from xai.explanation import xMIL


def get_additive_mil_model(input_dim=1024, num_classes=2, hidden_dim=256, device="cpu"):
    model = DefaultMILGraph(
        pointer=DefaultAttentionModule(
            hidden_activation=nn.LeakyReLU(0.2),
            hidden_dims=[hidden_dim, hidden_dim],
            input_dims=input_dim,
        ),
        classifier=AdditiveClassifier(
            hidden_dims=[hidden_dim, hidden_dim],
            input_dims=input_dim,
            output_dims=num_classes,
        ),
        num_classes=num_classes,
        device=device,
    )
    return model


class DefaultMILGraph(torch.nn.Module):

    def __init__(
        self,
        classifier: torch.nn.Module,
        pointer: torch.nn.Module,
        num_classes: int,
        device: str,
    ):
        super().__init__()
        self.classifier = classifier
        self.pointer = pointer
        self.n_classes = num_classes
        self.patch_logits = None
        self.attention_scores = None
        self.device = device

    def forward_fn(self, features, bag_sizes):
        return self.forward(features, bag_sizes)

    def forward(self, features, bag_sizes):
        """
        :param features: (#patches, input_dim)
        :param bag_sizes: (#bags)
        :return: (#bags, num_classes / features_dim)
        """
        attention_scores = self.pointer(features, bag_sizes)
        cls_out_dict = self.classifier(features, attention_scores, bag_sizes)
        self.attention_scores = attention_scores
        self.patch_logits = cls_out_dict.get("patch_logits")
        return cls_out_dict["logits"]


class StableSoftmax(torch.nn.Module):

    def __init__(self, dim=0) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, inputs):
        return torch.nn.LogSoftmax(dim=self.dim)(inputs).exp()


class DefaultAttentionModule(torch.nn.Module):

    def __init__(
        self,
        input_dims: int,
        hidden_dims: Sequence[int] = (),
        hidden_activation: torch.nn.Module = nn.ReLU(),
        output_activation: torch.nn.Module = StableSoftmax(dim=0),
        use_batch_norm: bool = True,
        track_bn_stats: bool = True,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.use_batch_norm = use_batch_norm
        self.track_bn_stats = track_bn_stats
        self.model = self.build_model()

    def build_model(self):
        nodes_by_layer = [self.input_dims] + list(self.hidden_dims) + [1]
        layers = []
        iterable = enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:]))
        for i, (nodes_in, nodes_out) in iterable:
            layer = nn.Linear(in_features=nodes_in, out_features=nodes_out, bias=True)
            layers.append(layer)
            if i < len(self.hidden_dims):
                if self.use_batch_norm:
                    layers.append(
                        nn.BatchNorm1d(
                            nodes_out, track_running_stats=self.track_bn_stats
                        )
                    )
                layers.append(self.hidden_activation)
        model = nn.Sequential(*layers)
        return model

    def bag_activation(self, scores, bag_sizes):
        scores_softmax = []
        for idx in range(len(bag_sizes)):
            bag_attention = self.output_activation(
                scores[bag_sizes[:idx].sum() : bag_sizes[: idx + 1].sum()]
            )
            scores_softmax.append(bag_attention)
        return torch.concat(scores_softmax, dim=0)

    def forward(self, features, bag_sizes):
        out = self.model(features)
        attention = self.bag_activation(out, bag_sizes)
        return attention


class AdditiveClassifier(torch.nn.Module):

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: Sequence[int] = (),
        hidden_activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.hidden_activation = hidden_activation
        self.model = self.build_model()

    def build_model(self):
        nodes_by_layer = [self.input_dims] + list(self.hidden_dims) + [self.output_dims]
        layers = []
        iterable = enumerate(zip(nodes_by_layer[:-1], nodes_by_layer[1:]))
        for i, (nodes_in, nodes_out) in iterable:
            layer = torch.nn.Linear(in_features=nodes_in, out_features=nodes_out)
            layers.append(layer)
            if i < len(self.hidden_dims):
                layers.append(self.hidden_activation)
        model = torch.nn.Sequential(*layers)
        return model

    @staticmethod
    def aggregate_patch_scores(patch_scores, bag_sizes):
        res = []
        for idx in range(len(bag_sizes)):
            patches_probs = patch_scores[
                bag_sizes[:idx].sum() : bag_sizes[: idx + 1].sum()
            ]  # n_patch x n_class
            bag_probs = patches_probs.sum(dim=0, keepdims=True)  # 1 x n_class
            res.append(bag_probs)
        return torch.concat(res, dim=0)

    def forward(self, features, attention, bag_sizes):
        attended_features = attention * features
        patch_logits = self.model(attended_features)
        logits = self.aggregate_patch_scores(patch_logits, bag_sizes)
        classifier_out_dict = {}
        classifier_out_dict["logits"] = logits
        classifier_out_dict["patch_logits"] = patch_logits
        return classifier_out_dict


class xAdditiveMIL(xMIL):

    def __init__(self, model, explained_class=None):
        super().__init__()
        self.model = model
        self.device = model.device
        self.explained_class = explained_class

    def attention_map(self, batch):
        """
        Returns the attention scores for the patches when the model is applied on the input data. The scores are
        softmaxed within each slide.
        """
        self.model.eval()
        features, bag_sizes = batch["features"].to(self.device), batch["bag_size"].to(
            self.device
        )
        self.model.forward(features, bag_sizes)
        return self.model.attention_scores.detach().cpu().numpy().squeeze()

    def explain_patch_scores(self, batch):
        """
        Returns the patch logits for the explained class before aggregation.
        """
        self.model.eval()
        features, bag_sizes = batch["features"].to(self.device), batch["bag_size"].to(
            self.device
        )
        self.model.forward(features, bag_sizes)
        explained_class = self.set_explained_class(self.explained_class)
        return (
            self.model.patch_logits[:, explained_class].detach().cpu().numpy().squeeze()
        )
