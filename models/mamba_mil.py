import torch
import torch.nn as nn

from captum.attr import IntegratedGradients

from models.simple_mamba import SimpleMamba
from models.bidirectional_mamba import BidirectionalMamba
from models.cross_mamba import CrossMamba

MAMBA = {
    "simple": SimpleMamba,
    "bidirectional": BidirectionalMamba,
    "cross": CrossMamba,
}

from xai.explanation import xMIL
from xai.lrp_rules import modified_linear_layer
from xai.lrp_utils import (
    var_data_requires_grad,
    set_detach_norm,
    set_lrp_params,
    layer_norm,
)


class MambaBlock(nn.Module):
    def __init__(self, features_dim=512, num_layers=1, scan="simple"):
        super().__init__()

        self.features_dim = features_dim
        self.num_layers = num_layers

        assert scan in [
            "simple",
            "bidirectional",
            "cross",
        ], "Only simple, bidirectional and cross scan are supported."
        self.scan = scan

        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(self.features_dim),
                        "mamba": MAMBA[self.scan](
                            d_model=self.features_dim,
                            d_state=16,
                            d_conv=4,
                            expand=2,
                        ),
                    }
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, h, positions_x=None, positions_y=None):
        h_ = h

        for layer in self.layers:
            h = layer["norm"](h)
            if self.scan == "cross":
                h = layer["mamba"](h, positions_x, positions_y)
            else:
                h = layer["mamba"](h)

        h = h + h_

        return h

    def xforward(self, h, detach_norm=None, lrp_params=None):
        detach_norm = set_detach_norm(detach_norm)
        lrp_params = set_lrp_params(lrp_params)

        h_ = h

        for layer in self.layers:
            if detach_norm is None:
                h = layer["norm"](h)
            else:
                modified_norm = layer_norm(
                    detach_norm=detach_norm,
                    weight=layer["norm"].weight,
                    bias=layer["norm"].bias,
                    dim=self.features_dim,
                )
                h = modified_norm(h)

            h = layer["mamba"].xforward(h, lrp_params=lrp_params)

        h = h + h_

        return h


class MambaMILModel(nn.Module):
    def __init__(
        self,
        input_dim,
        head_dim,
        device,
        features_dim=512,
        num_layers=1,
        pos_embed="none",
        scan="simple",
        is_survival=False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.head_dim = head_dim

        self.device = device

        self.features_dim = features_dim
        self.num_layers = num_layers

        assert pos_embed in [
            "none",
            "sinusoidal",
            "learned",
        ], "Only no, sinusoidal and learned positional embedding are supported."
        self.pos_embed = pos_embed

        assert scan in [
            "simple",
            "bidirectional",
            "cross",
        ], "Only simple, bidirectional and cross scan are supported."
        self.scan = scan

        self.is_survival = is_survival

        ## Projection

        self.proj = nn.Sequential(
            nn.Linear(
                self.input_dim,
                self.features_dim,
            ),
            nn.ReLU(),
        )

        if self.pos_embed == "learned":
            ## Learned Positonal Encoding

            self.pos_embed_x = nn.Parameter(torch.randn(1024, self.features_dim))
            self.pos_embed_y = nn.Parameter(torch.randn(1024, self.features_dim))

        ## Correlation

        self.mamba_block = MambaBlock(
            features_dim=self.features_dim, num_layers=self.num_layers, scan=self.scan
        )

        ## Layer Norm

        self.norm = nn.LayerNorm(self.features_dim)

        ## Aggregation

        self.attention = nn.Sequential(
            nn.Linear(self.features_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.attention_scores = None

        ## Linear Head

        if self.head_dim is not None:
            self.linear_head = nn.Linear(self.features_dim, self.head_dim)
        else:
            self.linear_head = None

    def softmax_attention_scores(self, attention_scores):
        attention_weights = nn.functional.softmax(
            attention_scores, dim=1
        )  # softmax along the sequence length dimension

        return attention_weights

    def aggregate(self, x, attention_scores):
        attention_weights = self.softmax_attention_scores(attention_scores)

        attention_weights = torch.transpose(
            attention_weights, 1, 2
        )  # [B, n, 1] -> [B, 1, n]

        x = torch.bmm(attention_weights, x)  # [B, 1, n]@[B, n, d] = [B, 1, d]
        x = x.squeeze(1)  # [B, d]

        return x

    def forward(self, h, positions_x=None, positions_y=None):
        B = h.shape[0]
        n = h.shape[1]

        ## Projection

        h = self.proj(h)  # [B, n, d]

        ## Positional Embedding

        if self.pos_embed == "sinusoidal":
            # Sinusoidal Positional Embedding

            pos_x = positions_x.unsqueeze(-1)  # [B, n] -> [B, n, 1]
            pos_y = positions_y.unsqueeze(-1)  # [B, n] -> [B, n, 1]

            omega = 1.0 / (
                10000.0
                ** (
                    torch.arange(self.features_dim // 4, device=self.device)
                    / (self.features_dim // 4 - 1)
                )
            )

            pos_x = pos_x * omega.view(
                1, 1, -1
            )  # [B, n, 1] * [1, 1, d // 4] = [B, n, d // 4]
            pos_y = pos_y * omega.view(
                1, 1, -1
            )  # [B, n, 1] * [1, 1, d // 4] = [B, n, d // 4]

            pe = torch.cat(
                (pos_x.sin(), pos_x.cos(), pos_y.sin(), pos_y.cos()), dim=-1
            )  # [B, n, d]

            h = h + pe
        elif self.pos_embed == "learned":
            # Learned Positional Embedding

            pe = self.pos_embed_x[positions_x] + self.pos_embed_y[positions_y]
            h = h + pe

        ## Correlation

        h = self.mamba_block(h, positions_x, positions_y)

        ## Layer Norm

        h = self.norm(h)

        ## Aggregation

        self.attention_scores = self.attention(h)  # [B, n, 1]
        h = self.aggregate(h, self.attention_scores)  # [B, d]

        ## Linear Head

        if self.linear_head is not None:
            res = self.linear_head(h)  # [B, head_dim]
        else:
            res = h

        if self.is_survival:
            hazards = torch.sigmoid(res)
            survivals = torch.cumprod(1 - hazards, dim=1)
            risk_score = -(torch.sum(survivals, dim=1))
            return hazards, survivals, risk_score
        else:
            return res

    def forward_fn(self, features, bag_sizes=None):
        return self.forward(features.to(self.device))

    def get_linear_head(self):
        return self.classifier

    def set_linear_head(self, linear_layer=None):
        self.classifier = linear_layer

    def get_out_layers(self):
        return None

    def set_out_layers(self, out_layers):
        pass

    def activations(
        self,
        h,
        detach_attn=True,
        detach_norm=None,
        lrp_params=None,
    ):
        assert (
            self.pos_embed == "none" and self.scan == "simple"
        ), "Explanations are only supported for no positional embedding and simple scan."

        lrp_params = set_lrp_params(lrp_params)
        activations = {}

        ##  Projection

        proj_input = h
        proj_input_data = var_data_requires_grad(proj_input)

        activations["projection"] = {
            "input": proj_input,
            "input-data": proj_input_data,
            "input-p": None,
        }

        proj_output = self.proj(proj_input_data)

        modified_proj = modified_linear_layer(
            self.proj[0], gamma=lrp_params["gamma"], no_bias=lrp_params["no_bias"]
        )
        proj_output_p = modified_proj(proj_input_data)

        ## Correlation

        mamba_input = proj_output
        mamba_input_data = var_data_requires_grad(mamba_input)
        mamba_input_p = proj_output_p

        activations["correlation"] = {
            "input": mamba_input,
            "input-data": mamba_input_data,
            "input-p": mamba_input_p,
        }

        mamba_output = self.mamba_block.xforward(
            mamba_input_data, detach_norm=detach_norm, lrp_params=lrp_params
        )

        ## Norm

        norm_input = mamba_output
        norm_input_data = var_data_requires_grad(norm_input)
        norm_input_p = None

        activations["norm"] = {
            "input": norm_input,
            "input-data": norm_input_data,
            "input-p": norm_input_p,
        }

        if detach_norm is None:
            norm_output = self.norm(norm_input_data)
        else:
            modified_norm = layer_norm(
                detach_norm=detach_norm,
                weight=self.norm.weight,
                bias=self.norm.bias,
                dim=self.features_dim,
            )
            norm_output = modified_norm(norm_input_data)

        ## Aggregation

        attn_input_data = var_data_requires_grad(norm_output)

        self.attention_scores = self.attention(attn_input_data)

        if detach_attn:
            self.attention_scores = self.attention_scores.detach()

        agg_input = norm_output
        agg_input_data = var_data_requires_grad(agg_input)
        agg_input_p = None

        activations["aggregation"] = {
            "input": agg_input,
            "input-data": agg_input_data,
            "input-p": agg_input_p,
        }

        agg_output = self.aggregate(agg_input_data, self.attention_scores)

        ## Linear Head

        if self.linear_head is not None:
            linear_head_input = agg_output
            linear_head_input_data = var_data_requires_grad(agg_output)
            linear_head_input_p = None

            activations["classification"] = {
                "input": linear_head_input,
                "input-data": linear_head_input_data,
                "input-p": linear_head_input_p,
            }

            linear_head_output = self.linear_head(linear_head_input_data)

            modified_linear_head = modified_linear_layer(
                self.linear_head,
                gamma=lrp_params["gamma"],
                no_bias=lrp_params["no_bias"],
            )
            linear_head_output_p = modified_linear_head(linear_head_input_data)

            ## Output

            activations["out"] = {
                "input": linear_head_output,
                "input-p": linear_head_output_p,
            }
        else:
            activations["out"] = {
                "input": agg_output,
                "input-p": None,
            }

        return activations


class xMambaMIL(xMIL):
    def __init__(
        self,
        model,
        head_type="classification",
        explained_class=None,
        explained_rel="logit",
        lrp_params=None,
        contrastive_class=None,
        detach_attn=True,
        detach_norm=None,
        model_restructuring=False,
    ):
        super().__init__(head_type)
        self.model = model
        self.device = model.device

        self.explained_class = explained_class
        self.explained_rel = explained_rel
        self.lrp_params = set_lrp_params(lrp_params)
        self.contrastive_class = contrastive_class
        self.detach_attn = detach_attn
        self.detach_norm = set_detach_norm(detach_norm)
        self.model_restructuring = model_restructuring

    def _get_prediction_score(self, features, batch):
        if self.model.is_survival:
            _, _, risk_scores = self.model(features)
            preds = risk_scores[0]
        else:
            logits = self.model(features)
            preds = logits[0, self.set_explained_class(batch)]
        return preds

    def attention_map(self, batch):
        self.model.eval()
        features = batch["features"].float().to(self.device)

        self.model.forward(features)

        attention_weights = self.model.softmax_attention_scores(
            self.model.attention_scores
        )

        return attention_weights.detach().cpu().numpy().squeeze()

    def explain_lrp(self, batch, verbose=False):
        self.model.eval()
        features = batch["features"].float().to(self.device)

        activations = self.model.activations(
            features,
            detach_attn=self.detach_attn,
            detach_norm=self.detach_norm,
            lrp_params=self.lrp_params,
        )

        bag_relevance, R = self.lrp_gi(
            activations=activations,
            explained_class=self.set_explained_class(batch),
            contrastive_class=self.contrastive_class,
            explained_rel=self.explained_rel,
            eps=self.lrp_params["eps"],
            verbose=False,
        )

        return bag_relevance.squeeze(), R, activations

    def explain_gi(self, batch):
        self.model.eval()
        features = batch["features"].float().to(self.device)
        features.requires_grad_(True)
        preds = self._get_prediction_score(features, batch)
        explanations_bag, explanations_vector = self.gradient_x_input(features, preds)
        return explanations_bag, explanations_vector

    def explain_squared_grad(self, batch):
        self.model.eval()
        features = batch["features"].float().to(self.device)
        features.requires_grad_(True)
        preds = self._get_prediction_score(features, batch)
        explanations_bag, explanations_vector = self.squared_grad(features, preds)
        return explanations_bag, explanations_vector

    def explain_integrated_gradients(self, batch):
        self.model.eval()
        features = batch["features"].float().to(self.device)
        if self.model.is_survival:
            ig = IntegratedGradients(lambda x: self.model(x)[-1])
            explanations_bag, explanations_vector = self.integrated_gradients(
                ig, features, None
            )
        else:
            ig = IntegratedGradients(lambda x: self.model(x))
            explanations_bag, explanations_vector = self.integrated_gradients(
                ig, features, self.set_explained_class(batch)
            )
        return explanations_bag, explanations_vector
