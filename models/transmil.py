"""
(c) The classes TransLayer, PPEG, TransMIL are partly copied from the repository of original transmil paper.

"""

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from captum.attr import IntegratedGradients

from models.utils import BlockLinear
from models.attention import Attention

from xai.lrp_rules import modified_linear_layer
from xai.lrp_utils import (
    var_data_requires_grad,
    set_detach_norm,
    set_lrp_params,
    layer_norm,
)
from xai.explanation import xMIL


class TransLayer(nn.Module):

    def __init__(
        self,
        norm_layer=nn.LayerNorm,
        dim=512,
        dropout_att=0.1,
        attention="nystrom",
        residual=True,
        heads=8,
        bias=True,
    ):
        super().__init__()
        if attention not in ["nystrom", "dot_prod"]:
            raise ValueError(
                "Only Nystrom and dot product attention can be used. "
                "Set attention method to 'nystrom' or 'dot_prod'"
            )
        self.norm = norm_layer(dim)
        self.attention = attention
        self.residual = residual
        self.heads = heads
        self.bias = bias

        self.attn = Attention(
            dim=dim,
            dim_head=dim // heads,
            heads=heads,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=residual,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=dropout_att,
            method=attention,
            bias=bias,
        )

    def forward(self, x, save_attn=False):
        feat_att = self.attn(self.norm(x), save_attn=save_attn)
        x = x + feat_att
        return x

    def xforward(self, x, detach_norm=None, lrp_params=None, verbose=False):
        """
        forward method for the explanation stage.

        :param x:
        :param detach_norm: [Dictionary or None (default)] dic containing booleans whether to detach the mean
             and/or the std in the normalization layer. None is equivalent to {'mean': False, 'std': False}.
        :param lrp_params: [Dictionary or None (default)] dic containing the necessary LRP parameters. None is
            equivalent to {'gamma': 0, 'eps': 1e-5, 'no_bias': False}. no_bias is True if the bias is discarded in
            LRP rules.
        :param verbose:
        :return:
        """
        detach_norm = set_detach_norm(detach_norm)
        lrp_params = set_lrp_params(lrp_params)

        if detach_norm is None:
            x_norm = self.norm(x)
        else:
            norm = layer_norm(
                detach_norm=detach_norm,
                weight=self.norm.weight,
                bias=self.norm.bias,
                dim=x.shape[-1],
                verbose=verbose,
            )
            x_norm = norm(x)

        feat_att = self.attn.xforward(x_norm, lrp_params=lrp_params, verbose=verbose)
        out = x + feat_att
        return out


class PPEG(nn.Module):
    def __init__(self, dim=512, cls_token=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)
        self.cls_token = cls_token

    def forward(self, x, H, W, detach_pe=False):
        B, _, C = x.shape
        if self.cls_token:
            cls_token, feat_token = x[:, 0], x[:, 1:]
        else:
            feat_token = x

        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if detach_pe:
            x = (
                cnn_feat
                + self.proj(cnn_feat).data
                + self.proj1(cnn_feat).data
                + self.proj2(cnn_feat).data
            )
        else:
            x = (
                cnn_feat
                + self.proj(cnn_feat)
                + self.proj1(cnn_feat)
                + self.proj2(cnn_feat)
            )
        x = x.flatten(2).transpose(1, 2)

        if self.cls_token:
            x = torch.cat((cls_token.unsqueeze(1), x), dim=1)

        return x

    def xforward(self, x, H, W, lrp_params, detach_pe=False):
        B, _, C = x.shape
        if self.cls_token:
            cls_token, feat_token = x[:, 0], x[:, 1:]
        else:
            feat_token = x

        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)

        proj1 = modified_linear_layer(
            self.proj, lrp_params["gamma"], lrp_params["no_bias"]
        )
        proj2 = modified_linear_layer(
            self.proj1, lrp_params["gamma"], lrp_params["no_bias"]
        )
        proj3 = modified_linear_layer(
            self.proj2, lrp_params["gamma"], lrp_params["no_bias"]
        )
        if detach_pe:
            x = (
                cnn_feat
                + proj1(cnn_feat).data
                + proj2(cnn_feat).data
                + proj3(cnn_feat).data
            )
        else:
            x = cnn_feat + proj1(cnn_feat) + proj2(cnn_feat) + proj3(cnn_feat)

        x = x.flatten(2).transpose(1, 2)

        if self.cls_token:
            x = torch.cat((cls_token.unsqueeze(1), x), dim=1)

        return x


class TransMILPooler(nn.Module):
    def __init__(self, method="cls_token", cls_token_ind=0):
        """
        method == 'cls_token': the pooling is done only by taking the first token as the class token.
        method == 'sum'
        """
        super().__init__()

        self.method = method
        if method == "cls_token":
            self.pooler = lambda x: x[:, cls_token_ind]
        elif method == "sum":
            self.pooler = lambda x: x.sum(dim=1)
        else:
            raise NotImplementedError(f"Pooling method '{method}' is not implemented.")

    def forward(self, x):
        return self.pooler(x)


class TransMIL(nn.Module):
    def __init__(
        self,
        n_feat_input,
        n_feat,
        head_dim,
        device,
        attention="nystrom",
        n_layers=2,
        dropout_att=0.1,
        dropout_class=0.5,
        dropout_feat=0,
        attn_residual=True,
        pool_method="cls_token",
        n_out_layers=0,
        bias=True,
        num_encoders=1,
        is_survival=False,
        use_ppeg=True,
    ):
        """

        :param n_feat_input: (int) Dimension of the incoming feature vectors.
        :param n_feat: (int) Output dimension of the linear layer applied to the feature vectors.
        :param head_dim: (int) output dimension of the last linear layer.
                                In a classification task, it is number of classes.
                                In a regression task, it should be 1.
                                In survival task, it is the number of time intervals.
        :param device: the operating device
        :param attention: (str) attention type. can be 'nystrom' or 'dot_prod'
        :param n_layers: (int) number of transformer layers
        :param dropout_att: (float) probability of features after the self-attention to be zeroed. Default: 0
        :param dropout_class: (float) probability of features before the classification to be zeroed. Default: 0
        :param dropout_feat: (float) probability of features after the linear layers to be zeroed. Default: 0
        :param attn_residual: (bool) if True, there will be a residual connection in self attention. default: True.
        :param pool_method: (str) can be 'cls_token' (default) or 'sum'.
        :param n_out_layers: (int) number of linear layers applied before the classifier.
        :param bias: (bool) if False then the bias term is omited from all linear layers. default: True
        :param num_encoders: (int): number of encoders at the first layer
        :param is_survival: (bool) whether the model is learning a survival task
        :param use_ppeg: (bool) if true, there will be a PPEG layer
        """
        super().__init__()
        if n_layers < 2:
            raise ValueError(
                f"Number of transformer layers should be at least 2, n_layers={n_layers} given."
            )

        self.bias = bias
        self.n_feat_input = n_feat_input
        self.n_feat = n_feat
        self.head_dim = head_dim
        self.device = device
        self.n_layers = n_layers
        self.num_encoders = num_encoders
        self.is_survival = is_survival
        self.use_ppeg = use_ppeg

        if self.num_encoders == 1:
            encoder_layer = nn.Linear
        else:
            encoder_layer = partial(BlockLinear, num_blocks=num_encoders)

        self._fc1 = nn.Sequential(
            encoder_layer(n_feat_input, n_feat, bias=bias), nn.ReLU()
        )
        self.pos_layer = (
            PPEG(dim=n_feat, cls_token=(pool_method == "cls_token"))
            if use_ppeg
            else None
        )
        self.norm = nn.LayerNorm(n_feat)
        self.attention = attention
        self.translayers = nn.Sequential(
            *[
                TransLayer(
                    dim=n_feat,
                    dropout_att=dropout_att,
                    attention=attention,
                    residual=attn_residual,
                    bias=bias,
                )
                for _ in range(n_layers)
            ]
        )

        self.dropout_class = nn.Dropout(dropout_class)
        self.dropout_feat = nn.Dropout(dropout_feat)

        self.pool_method = pool_method
        self.cls_token = nn.Parameter(torch.randn(1, 1, n_feat))

        self.pooler = TransMILPooler(method=pool_method)

        # MLP at output
        mlp_layers = [
            nn.Sequential(
                nn.Linear(n_feat, n_feat, bias=bias),
                nn.ReLU(),
            )
            for _ in range(n_out_layers)
        ]
        self.mlp_layers = nn.Sequential(*mlp_layers)

        if head_dim is not None:
            self._fc2 = nn.Linear(n_feat, head_dim, bias=bias)
        else:
            self._fc2 = None

    def _pad(self, h):
        """
        pads the input with part of itself so that the last dimension is a power of 2. Then it adds a random
        token to at the beginning of each slide as the class token.

        """
        H = h.shape[1]
        H_ = int(np.ceil(np.sqrt(H)))
        add_length = H_**2 - H
        cat_h = h[:, :add_length, :]
        h = torch.cat([h, cat_h], dim=1)  # [B, N, n_feat]
        return h, H_

    def _add_clstoken(self, h):
        if self.pool_method != "cls_token":
            raise ValueError(
                "The class token can be added only if the cls_token argument of the model is True."
            )
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(self.device)
        return torch.cat((cls_tokens, h), dim=1)

    def forward(self, x, detach_pe=False, save_attn=False):
        h = x.float()  # [B, n, n_feat_input]
        h = self._fc1(h)  # [B, n, n_feat]
        h = self.dropout_feat(h)

        # ----> pad
        if self.use_ppeg:
            h, _H = self._pad(h)

        # ---->cls_token
        if self.pool_method == "cls_token":
            h = self._add_clstoken(h)

        # ---->Translayer x1
        h = self.translayers[0](h, save_attn=save_attn)

        # ---->PPEG
        if self.use_ppeg:
            h = self.pos_layer(h, _H, _H, detach_pe)  # [B, N, n_feat]

        # ---->Translayer x2 onwards
        for layer in self.translayers[1:]:
            h = layer(h, save_attn=save_attn)

        h = self.norm(h)

        # ----> notmalize and pool
        if self.pool_method == "cls_token":
            h = self.pooler(h)  # [B, n_feat]

        if self.mlp_layers:
            h = self.mlp_layers(h)
        # ---->predict
        h = self.dropout_class(h)
        if self._fc2 is not None:
            res = self._fc2(h)  # [B, n_classes]

            if self.pool_method == "sum":
                res = self.pooler(res)  # [B, n_feat]

        else:
            res = h

        if self.is_survival:
            hazards = torch.sigmoid(res)
            survivals = torch.cumprod(1 - hazards, dim=1)
            risk_score = -(torch.sum(survivals, dim=1))
            return hazards, survivals, risk_score

        return res

    def forward_fn(self, features, bag_sizes=None):
        return self.forward(features)

    def set_linear_head(self, linear_layer=None):
        self._fc2 = linear_layer

    def get_linear_head(self):
        return self._fc2

    def get_out_layers(self):
        return self.mlp_layers

    def set_out_layers(self, out_layers):
        self.mlp_layers = out_layers

    def activations(
        self, x, detach_norm=None, detach_pe=False, lrp_params=None, verbose=False
    ):
        """
        method for collecting the activations for the explanation stage.

        Args:
            x: [n_batch x n_patch x n_feat] input feature tensor.
            detach_norm: [Dictionary or None (default)] dict containing booleans whether to detach the mean
             and/or the std in the normalization layer. None is equivalent to {'mean': False, 'std': False}.

            detach_pe: (bool) if True the positional encoder will be detached from the computational graph when
            performing the xforward of PPEG

            lrp_params: [Dictionary or None (default)] dic containing the necessary LRP parameters. None is
            equivalent to {'gamma': 0, 'eps': 1e-5, 'no_bias': False}. no_bias is True if the bias is discarded in
            LRP rules.

        Returns:
            activations: [Dictionary] The keys are the name of the layers. for a given layer the value is a dic as
            {'input': ..., 'input-data': ..., 'input-p':...}. 'input' is the output of the prev layer, 'input-data' is
            'input' detached and then attached to the comp graph (see utils_lrp.var_data_requires_grad). 'input-p'
            is the output of the xforward method of the previous layer.

        """
        if self.num_encoders > 1:
            raise ValueError("Cannot compute LRP for BlockLinear layer yet.")

        if detach_norm is None:
            norm_last = self.norm
        else:
            norm_last = layer_norm(
                detach_norm=detach_norm,
                weight=self.norm.weight,
                bias=self.norm.bias,
                dim=x.shape[-1],
                verbose=verbose,
            )

        lrp_params = set_lrp_params(lrp_params)
        detach_norm = set_detach_norm(detach_norm)
        activations = {}

        # ----> feature reduction
        fc1_input = x
        fc1_input_data = var_data_requires_grad(fc1_input)
        activations["fc1"] = {
            "input": fc1_input,
            "input-data": fc1_input_data,
            "input-p": None,
        }

        feats = self._fc1(
            fc1_input_data
        )  # [B, n, n_feat_input] --> [B, n_patches, n_feat]

        _fc1_ = modified_linear_layer(
            self._fc1[0], lrp_params["gamma"], no_bias=lrp_params["no_bias"]
        )
        feats_p = _fc1_(fc1_input_data)

        # ----> pad and add cls token
        if self.use_ppeg:
            feats, _H = self._pad(feats)  # [B, N, n_feat]
        if self.pool_method == "cls_token":
            feats = self._add_clstoken(feats)

        if self.use_ppeg:
            feats_p, _H = self._pad(feats_p)  # [B, N, n_feat]
        if self.pool_method == "cls_token":
            feats_p = self._add_clstoken(feats_p)

        # ---->Translayer 0
        attn0_input = feats
        attn0_input_p = feats_p
        attn0_input_data = var_data_requires_grad(attn0_input)
        activations["translayer-0"] = {
            "input": attn0_input,
            "input-data": attn0_input_data,
            "input-p": attn0_input_p,
        }

        attn_output = self.translayers[0].xforward(
            attn0_input_data,
            detach_norm=detach_norm,
            lrp_params=lrp_params,
            verbose=verbose,
        )
        # ---->PPEG
        if self.use_ppeg:
            pos_enc_input = attn_output
            pos_enc_input_p = None
            pos_enc_input_data = var_data_requires_grad(pos_enc_input)
            activations["pos-enc"] = {
                "input": pos_enc_input,
                "input-data": pos_enc_input_data,
                "input-p": pos_enc_input_p,
            }

            pos_enc_output = self.pos_layer.xforward(
                pos_enc_input_data, _H, _H, lrp_params, detach_pe
            )

        # ---->Translayer 1 onwards
        if self.use_ppeg:
            attn_input = pos_enc_output
        else:
            attn_input = attn_output
        attn_input_p = None

        for i_layer, layer in enumerate(self.translayers[1:]):
            attn_input_data = var_data_requires_grad(attn_input)
            activations[f"translayer-{i_layer + 1}"] = {
                "input": attn_input,
                "input-data": attn_input_data,
                "input-p": attn_input_p,
            }
            attn_output = layer.xforward(
                attn_input_data,
                detach_norm=detach_norm,
                lrp_params=lrp_params,
                verbose=verbose,
            )

            attn_input = attn_output
            attn_input_p = None

        # ----> layernorm
        norm_input = attn_input
        norm_input_p = attn_input_p
        norm_input_data = var_data_requires_grad(norm_input)
        activations["norm-layer"] = {
            "input": norm_input,
            "input-data": norm_input_data,
            "input-p": norm_input_p,
        }

        norm_output = norm_last(norm_input_data)  # [B, n_feat]

        # ----> pooler
        if self.pool_method == "cls_token":
            pooler_input = norm_output
            pooler_input_p = None
            pooler_input_data = var_data_requires_grad(pooler_input)
            activations["pooler"] = {
                "input": pooler_input,
                "input-data": pooler_input_data,
                "input-p": pooler_input_p,
            }

            pooler_output = self.pooler(pooler_input_data)  # [B, n_feat]

        # ----> mlp layers
        if self.pool_method == "cls_token":
            mlp_input = pooler_output
        else:
            mlp_input = norm_output

        mlp_input_p = None

        for i_layer, layer in enumerate(self.mlp_layers):
            fc = layer[0]

            mlp_input_data = var_data_requires_grad(mlp_input)
            activations[f"mlp-{i_layer}"] = {
                "input": mlp_input,
                "input-data": mlp_input_data,
                "input-p": mlp_input_p,
            }

            fc_ = modified_linear_layer(
                fc, lrp_params["gamma"], no_bias=lrp_params["no_bias"]
            )
            mlp_out = F.relu(fc(mlp_input_data))
            mlp_out_p = fc_(mlp_input_data)

            mlp_input = mlp_out
            mlp_input_p = mlp_out_p

        # ---->predict
        if self._fc2 is not None:
            classifier_input = mlp_input
            classifier_input_p = mlp_input_p
            classifier_input_data = var_data_requires_grad(classifier_input)
            activations["classifier"] = {
                "input": classifier_input,
                "input-data": classifier_input_data,
                "input-p": classifier_input_p,
            }

            logits = self._fc2(classifier_input_data)  # [B, n_classes]

            classifier_ = modified_linear_layer(
                self._fc2, lrp_params["gamma"], no_bias=lrp_params["no_bias"]
            )
            logits_p = classifier_(classifier_input_data)

            if self.pool_method == "sum":
                pooler_input = logits
                pooler_input_p = logits_p
                pooler_input_data = var_data_requires_grad(pooler_input)
                activations["pooler"] = {
                    "input": pooler_input,
                    "input-data": pooler_input_data,
                    "input-p": pooler_input_p,
                }

                pooler_output = self.pooler(pooler_input_data)  # [B, n_feat]

                activations["out"] = {"input": pooler_output, "input-p": None}
            else:
                activations["out"] = {"input": logits, "input-p": logits_p}

        else:
            activations["out"] = {"input": norm_output, "input-p": None}

        return activations

    def set_attentions_to_none(self):
        for layer in self.translayers:
            layer.attn.attn_scores = None


class xTransMIL(xMIL):
    """
    class for generating explanation heatmaps for a given TransMIL model and an input.
    possible methods are:
                        attention: Attention rollout (attention map)
                        lrp : LRP
                        gi : Gradient x Input
                        grad2: squared grandient

    method get_heatmap(batch, heatmap_type) from the base class can be used to get the heatmap of desired method.
    """

    def __init__(
        self,
        model,
        head_type="classification",
        explained_class=None,
        explained_rel="logit",
        lrp_params=None,
        contrastive_class=None,
        discard_ratio=0,
        attention_layer=None,
        head_fusion="mean",
        detach_norm=None,
        detach_mean=False,
        detach_pe=False,
        use_ppeg=True,
    ):
        """
        Args:
            explained_class: 0 or 1, or None. if None, the target class is explained

            explained_rel: the output relevance. can be:
                'logit-diff': the difference of the logits (1st eq. of p. 202 of Montavon et. al., 2019)
                'logits': the logits without any change

            lrp_params: [Dictionary or None (default)] dic containing the necessary LRP parameters. None is
                equivalent to {'gamma': 0, 'eps': 1e-5, 'no_bias': False}. no_bias is True if the bias is discarded in
                LRP rules.

            contrastive_class

            attention_layer: [int] The layer from which to extract attention scores. If None, all layers are
                multiplied (attention rollout).

            detach_norm: [Dictionary or None (default)] dic containing booleans whether to detach the mean
             and/or the std in the normalization layer. None is equivalent to {'mean': False, 'std': False}.

            detach_pe: (bool) if True the positional encoder will be detached from the computational graph when
            performing the xforward of PPEG

            use_ppeg: (bool) if True, there will be a PPEG layer
        """
        super().__init__(head_type)
        self.model = model
        self.device = model.device
        self.explained_class = explained_class
        self.explained_rel = explained_rel
        self.lrp_params = set_lrp_params(lrp_params)
        self.contrastive_class = contrastive_class
        self.discard_ratio = discard_ratio
        self.attention_layer = attention_layer
        self.head_fusion = head_fusion
        self.detach_norm = set_detach_norm(detach_norm)
        self.detach_mean = detach_mean
        self.detach_pe = detach_pe
        self.use_ppeg = use_ppeg

    def _get_prediction_score(self, features, batch):
        if self.model.is_survival:
            _, _, risk_scores = self.model(features, self.detach_pe)
            preds = risk_scores[0]
        else:
            logits = self.model(features, self.detach_pe)
            preds = logits[0, self.set_explained_class(batch)]
        return preds

    def attention_map(self, batch):
        """
        Attention Rollout method from https://arxiv.org/abs/2005.00928
        (c) minimally modified version of https://github.com/jacobgil/vit-explain/tree/main
        """
        n_patches = batch["bag_size"].item()
        features, bag_sizes, targets = (
            batch["features"],
            batch["bag_size"],
            batch["targets"],
        )
        features = features.to(torch.float32).to(self.device)

        self.model.eval()
        _ = self.model(features, self.detach_pe, save_attn=True)

        n_attention_tokens = self.model.translayers[0].attn.attn_scores.shape[-2]
        if self.use_ppeg:
            square_pad_tokens = int(np.ceil(np.sqrt(n_patches))) ** 2 - n_patches
            attn_pad_tokens = n_attention_tokens - (1 + n_patches + square_pad_tokens)

        if self.attention_layer is not None:
            attention = self.model.translayers[
                self.attention_layer
            ].attn.attn_scores.detach()
            if self.head_fusion == "mean":
                result = attention.mean(axis=1)
            elif self.head_fusion == "max":
                result = attention.max(axis=1)[0]
            elif self.head_fusion == "min":
                result = attention.min(axis=1)[0]
            else:
                raise f"Attention head fusion type not supported: {self.head_fusion}"
        else:
            # Attention rollout
            result = torch.eye(n_attention_tokens).to(self.device)
            with torch.no_grad():
                for layer in self.model.translayers:
                    attention = layer.attn.attn_scores.detach()
                    if self.head_fusion == "mean":
                        attention_heads_fused = attention.mean(axis=1)
                    elif self.head_fusion == "max":
                        attention_heads_fused = attention.max(axis=1)[0]
                    elif self.head_fusion == "min":
                        attention_heads_fused = attention.min(axis=1)[0]
                    else:
                        raise f"Attention head fusion type not supported: {self.head_fusion}"

                    # Drop the lowest attentions, but
                    # don't drop the class token
                    flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                    _, indices = flat.topk(
                        int(flat.size(-1) * self.discard_ratio), -1, False
                    )
                    indices = indices[indices != 0]
                    flat[0, indices] = 0

                    I = torch.eye(attention_heads_fused.size(-1)).to(self.device)
                    a = (attention_heads_fused + 1.0 * I) / 2
                    a = a / a.sum(dim=-1)

                    result = torch.matmul(a, result)
        if self.use_ppeg:
            mask = result[
                0,
                attn_pad_tokens,
                attn_pad_tokens + 1 : attn_pad_tokens + 1 + n_patches,
            ]
        else:
            mask = result[
                0,
                0,
                1
                + self.model.translayers[0].attn.padding : 1
                + self.model.translayers[0].attn.padding
                + n_patches,
            ]
        self.model.set_attentions_to_none()
        return mask.cpu().detach().numpy()

    def explain_lrp(self, batch, verbose=False):
        """
        Method for using gradient x input for explanation.

        Returns:
            bag_relevance: [n_patches x 1] the relevance of each patch at the input space

            R: dictionary with the relevance at each layer (the keys are the layer names)

            activations: activations of the layers

        """
        if self.model._fc2 is None:
            raise NotImplementedError(
                "This explanation method can be currently only used for classification."
            )

        features = batch["features"].to(torch.float32).to(self.device)

        self.model.eval()
        activations = self.model.activations(
            features,
            detach_norm=self.detach_norm,
            detach_pe=self.detach_pe,
            lrp_params=self.lrp_params,
        )
        bag_relevance, R = self.lrp_gi(
            activations,
            self.set_explained_class(batch),
            self.contrastive_class,
            self.explained_rel,
            self.lrp_params["eps"],
            verbose,
        )

        return bag_relevance.squeeze(), R, activations

    def explain_gi(self, batch):
        self.model.eval()
        features = batch["features"].to(self.device)
        features.requires_grad_(True)
        preds = self._get_prediction_score(features, batch)
        explanations_bag, explanations_vector = self.gradient_x_input(features, preds)
        return explanations_bag, explanations_vector

    def explain_squared_grad(self, batch):
        self.model.eval()
        features = batch["features"].to(self.device)
        features.requires_grad_(True)
        preds = self._get_prediction_score(features, batch)
        explanations_bag, explanations_vector = self.squared_grad(features, preds)
        return explanations_bag, explanations_vector

    def explain_integrated_gradients(self, batch):
        self.model.eval()
        features = batch["features"].to(self.device)
        if self.model.is_survival:
            ig = IntegratedGradients(lambda x: self.model(x, self.detach_pe)[-1])
            explanations_bag, explanations_vector = self.integrated_gradients(
                ig, features, None
            )
        else:
            ig = IntegratedGradients(lambda x: self.model(x, self.detach_pe))
            explanations_bag, explanations_vector = self.integrated_gradients(
                ig, features, self.set_explained_class(batch)
            )
        return explanations_bag, explanations_vector
