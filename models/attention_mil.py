from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients

from models.utils import Classifier
from xai.lrp_rules import modified_linear_layer
from xai.lrp_utils import var_data_requires_grad, set_lrp_params
from xai.explanation import xMIL


class AttentionMILModel(nn.Module):

    def __init__(
        self,
        input_dim,
        num_classes=None,
        features_dim=256,
        inner_attention_dim=128,
        dropout=None,
        dropout_strategy="features",
        num_layers=1,
        n_out_layers=0,
        bias=True,
        device="cpu",
    ):
        """
        :param input_dim: (int) Dimension of the incoming feature vectors.
        :param num_classes: (int) Number of classes to predict.
        :param features_dim: (int) Output dimension of the linear layer applied to the feature vectors.
        :param inner_attention_dim: (int) Inner hidden dimension of the 2-layer attention mechanism.
        :param dropout: (float) Fraction of neurons to drop per targeted layer. None to apply no dropout.
        :param dropout_strategy: (str) Which layers to apply dropout to.
        :param num_layers: (int) number of linear layers applied to feature vectors.
        :param n_out_layers: (int) relevant for additive model. number of linear layers applied before the classifier
        to the features scaled by the attention values.
        :param bias: (bool) if False then the bias term is omited from all linear layers. default: True
        :param device: the operating device
        """
        super(AttentionMILModel, self).__init__()
        # Save args
        self.bias = bias
        self.input_dim = input_dim
        self.n_classes = num_classes
        self.features_dim = features_dim
        self.inner_attention_dim = inner_attention_dim
        self.num_layers = num_layers
        # Set up model
        layer1 = [
            nn.Sequential(nn.Linear(self.input_dim, self.features_dim), nn.ReLU())
        ]

        layer2_onwards = [
            nn.Sequential(
                nn.Linear(self.features_dim, self.features_dim, bias=bias),
                nn.ReLU(),
            )
            for _ in range(num_layers - 1)
        ]

        self.linear_layers = nn.Sequential(*(layer1 + layer2_onwards))

        self.attention = nn.Sequential(
            nn.Linear(self.features_dim, self.inner_attention_dim),
            nn.Tanh(),
            nn.Linear(self.inner_attention_dim, 1),
        )
        out_layers = [
            nn.Sequential(
                nn.Linear(self.features_dim, self.features_dim, bias=bias),
                nn.ReLU(),
            )
            for _ in range(n_out_layers)
        ]
        self.out_layers = nn.Sequential(*out_layers)

        self.classifier = nn.Linear(self.features_dim, self.n_classes, bias=bias)

        # Set up dropout layers
        self.feature_dropout, self.linear_dropout, self.classifier_dropout = (
            None,
            None,
            None,
        )
        if dropout is not None:
            if dropout_strategy == "features":
                self.feature_dropout = nn.Dropout(dropout)
            elif dropout_strategy == "all":
                self.feature_dropout = nn.Dropout(dropout)
                self.linear_dropout = nn.Dropout(dropout)
                self.classifier_dropout = nn.Dropout(dropout)
            elif dropout_strategy == "last":
                self.classifier_dropout = nn.Dropout(dropout)

        self.device = device

        self.attention_scores = None

    @staticmethod
    def softmax_scores_bag(scores, bag_sizes):
        scores_softmax = []
        for idx in range(len(bag_sizes)):
            bag_attention = torch.softmax(
                scores[bag_sizes[:idx].sum() : bag_sizes[: idx + 1].sum()], dim=0
            )
            scores_softmax.append(bag_attention)
        return torch.concat(scores_softmax, dim=0)

    def bag_aggregation(self, features, attention_scores, bag_sizes):
        """
        :param features: (#patches, features_dim)
        :param attention_scores: (#patches, 1)
        :param bag_sizes: (#bags)
        :return: (#bags, features_dim)
        """

        bag_embeddings = []
        for idx in range(len(bag_sizes)):
            bag_features = features[bag_sizes[:idx].sum() : bag_sizes[: idx + 1].sum()]
            bag_attention = torch.softmax(
                attention_scores[bag_sizes[:idx].sum() : bag_sizes[: idx + 1].sum()],
                dim=0,
            )
            bag_embeddings.append(
                torch.mm(torch.transpose(bag_attention, 0, 1), bag_features)
            )

        return torch.concat(bag_embeddings, dim=0)

    def aggregate_patch_scores(self, patch_scores, bag_sizes):
        res = []
        for idx in range(len(bag_sizes)):
            patches_probs = patch_scores[
                bag_sizes[:idx].sum() : bag_sizes[: idx + 1].sum()
            ]  # n_patch x n_class
            bag_probs = patches_probs.sum(dim=0, keepdims=True)  # 1 x n_class
            res.append(bag_probs)
        return torch.concat(res, dim=0)

    def forward(self, features, bag_sizes):
        """
        :param features: (#patches, input_dim)
        :param bag_sizes: (#bags)
        :return: (#bags, num_classes / features_dim)
        """
        # Apply pre-aggregation layers
        if self.feature_dropout is not None:
            features = self.feature_dropout(features)
        for block in self.linear_layers:
            features = block(features)
        if self.linear_dropout is not None:
            features = self.linear_dropout(features)

        # Apply attention aggregation
        self.attention_scores = self.attention(features)

        bag_embeddings = self.bag_aggregation(
            features, self.attention_scores, bag_sizes
        )

        if self.out_layers:
            bag_embeddings = self.out_layers(bag_embeddings)

        # Apply classifier
        if self.classifier_dropout is not None:
            bag_embeddings = self.classifier_dropout(bag_embeddings)

        res = self.classifier(bag_embeddings)

        return res

    def forward_fn(self, features, bag_sizes):
        return self.forward(features, bag_sizes)

    def activations(
        self, features, bag_sizes, detach_attn=True, lrp_params=None, verbose=False
    ):
        """
        method for collecting the activations for the explanation stage.

        Args:
            :param features: [(n_batch.n_patch) x n_feat] input feature tensor.
            :param bag_sizes: [n_batch]: a list including the n_patch for each sample of the batch
            :param detach_attn: (bool) If True, the self attention head is detached from the comp graph.
            :param lrp_params: [Dictionary or None (default)] dic containing the necessary LRP parameters. None is
                equivalent to {'gamma': 0, 'eps': 1e-5, 'no_bias': False}. no_bias is True if the bias is discarded in
                LRP rules.
            :param verbose: whether to print logs

        Returns:
            activations: [Dictionary] The keys are the name of the layers. for a given layer the value is a dictionary
                as {'input': ..., 'input-data': ..., 'input-p':...}. 'input' is the output of the prev layer,
                'input-data' is 'input' detached and then attached to the comp graph
                (see xai.lrp_utils.var_data_requires_grad). 'input-p' is the output of the xforward method of the
                previous layer.

        """

        lrp_params = set_lrp_params(lrp_params)
        activations = {}

        linear_input = features
        linear_input_p = None

        # Apply pre-aggregation layers
        for i_block, block in enumerate(self.linear_layers):
            linear_input_data = var_data_requires_grad(linear_input)
            activations[f"fc1-{i_block}"] = {
                "input": linear_input,
                "input-data": linear_input_data,
                "input-p": linear_input_p,
            }
            fc = block[0]
            fc_ = modified_linear_layer(
                fc, lrp_params["gamma"], no_bias=lrp_params["no_bias"]
            )
            block_out = F.relu(fc(linear_input_data))
            block_out_p = fc_(linear_input_data)

            linear_input = block_out
            linear_input_p = block_out_p

        # compute attention values
        attn_input_data = var_data_requires_grad(linear_input)
        self.attention_scores = self.attention(attn_input_data)
        if detach_attn:
            self.attention_scores = self.attention_scores.detach()
            if verbose:
                print("attention values were detached from the computational graph!")

        #  Apply attention aggregation
        agg_input = linear_input
        agg_input_p = linear_input_p
        agg_input_data = var_data_requires_grad(agg_input)
        activations["aggregation"] = {
            "input": agg_input,
            "input-data": agg_input_data,
            "input-p": agg_input_p,
        }
        agg_output = self.bag_aggregation(
            agg_input_data, self.attention_scores, bag_sizes
        )

        #  relevance for Additive model: apply linear layers after attention scaling
        out_layer_input = agg_output
        out_layer_input_p = None
        for i_layer, layer in enumerate(self.out_layers):
            fc = layer[0]

            out_layer_input_data = var_data_requires_grad(out_layer_input)
            activations[f"layerout-{i_layer}"] = {
                "input": out_layer_input,
                "input-data": out_layer_input_data,
                "input-p": out_layer_input_p,
            }

            fc_ = modified_linear_layer(
                fc, lrp_params["gamma"], no_bias=lrp_params["no_bias"]
            )
            mlp_out = F.relu(fc(out_layer_input_data))
            mlp_out_p = fc_(out_layer_input_data)

            out_layer_input = mlp_out
            out_layer_input_p = mlp_out_p

        # apply classifier
        classifier_input = out_layer_input
        classifier_input_p = out_layer_input_p
        classifier_input_data = var_data_requires_grad(classifier_input)
        activations[f"classifier"] = {
            "input": classifier_input,
            "input-data": classifier_input_data,
            "input-p": classifier_input_p,
        }
        logits = self.classifier(classifier_input_data)
        classifier_ = modified_linear_layer(
            self.classifier, lrp_params["gamma"], no_bias=lrp_params["no_bias"]
        )
        logits_p = classifier_(classifier_input_data)
        activations["out"] = {"input": logits, "input-p": logits_p}

        return activations


class xAttentionMIL(xMIL):
    """
    class for generating explanation heatmaps for a given AttentionMILModel model and an input.
    possible methods are:
                        attention: Attention rollout (attention map)
                        lrp : LRP
                        gi : Gradient x Input
                        grad2: squared grandient
                        perturbation_keep: perturbation based method for Early et al 2022.

    method get_heatmap(batch, heatmap_type) from the base class can be used to get the heatmap of desired method.
    """

    def __init__(
        self,
        model,
        explained_class=None,
        explained_rel="logit",
        lrp_params=None,
        contrastive_class=None,
        detach_attn=True,
    ):
        super().__init__()
        self.model = model
        self.device = model.device
        self.explained_class = explained_class
        self.lrp_params = set_lrp_params(lrp_params)
        self.explained_rel = explained_rel
        self.contrastive_class = contrastive_class
        self.detach_attn = detach_attn

    def attention_map(self, batch):
        """
        returns the attention scores for the patches when the model is applied on the input data. The scores are
        softmaxed within each slide before returning themn.
        """
        self.model.eval()
        features, bag_sizes = batch["features"].to(self.device), batch["bag_size"].to(
            self.device
        )
        self.model.forward(features, bag_sizes)
        attention_scores = self.model.softmax_scores_bag(
            self.model.attention_scores, bag_sizes
        )
        return attention_scores.detach().cpu().numpy().squeeze()

    def explain_lrp(self, batch, verbose=False):
        if self.model.classifier is None:
            raise NotImplementedError()

        features, bag_sizes = batch["features"].to(self.device), batch["bag_size"].to(
            self.device
        )

        self.model.eval()
        activations = self.model.activations(
            features,
            bag_sizes,
            detach_attn=self.detach_attn,
            lrp_params=self.lrp_params,
            verbose=verbose,
        )
        bag_relevance, R = self.lrp_gi(
            activations,
            self.set_explained_class(batch),
            self.contrastive_class,
            self.explained_rel,
            self.lrp_params["eps"],
            verbose,
        )
        return bag_relevance, R, activations

    def explain_gi(self, batch):
        self.model.eval()
        features, bag_sizes = batch["features"].to(self.device), batch["bag_size"].to(
            self.device
        )
        features.requires_grad_(True)
        logits = self.model(features, bag_sizes)
        return self.gradient_x_input(
            features, logits[0, self.set_explained_class(batch)]
        )

    def explain_squared_grad(self, batch):
        self.model.eval()
        features, bag_sizes = batch["features"].to(self.device), batch["bag_size"].to(
            self.device
        )
        features.requires_grad_(True)
        logits = self.model(features, bag_sizes)
        return self.squared_grad(features, logits[0, self.set_explained_class(batch)])

    def explain_perturbation(self, batch, perturbation_method):
        def forward_fn(features, bag_sizes):
            features, bag_sizes = features.to(self.device), bag_sizes.to(self.device)
            return self.model(features, bag_sizes)

        self.model.eval()
        explained_class = self.set_explained_class(batch)
        return self.perturbation_scores(
            batch, perturbation_method, forward_fn, explained_class, self.explained_rel
        )

    def explain_integrated_gradients(self, batch):
        def forward_fn_(bag_sizes_, features_):
            return self.model(features_, bag_sizes_)

        self.model.eval()
        features, bag_sizes = batch["features"].to(self.device), batch["bag_size"].to(
            self.device
        )
        forward_ = partial(forward_fn_, bag_sizes)

        ig = IntegratedGradients(forward_)
        explanations = self.integrated_gradients(
            ig, features, self.set_explained_class(batch)
        )

        return explanations
