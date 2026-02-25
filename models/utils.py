import math

import torch
import torch.nn as nn

from models.loss import SurvivalNegLogLikelihood


class ModelEngine:

    def __init__(
        self,
        model,
        learning_rate,
        weight_decay,
        optimizer="SGD",
        loss_type="cross-entropy",
        gradient_clip=None,
        device=torch.device("cpu"),
    ):
        self.model = model

        if model.head_dim is None:
            raise ValueError(f"Cannot train a model if model.head_dim is None.")

        # Set up optimizer
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay,
            )
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                amsgrad=False,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        # Set up loss function
        if loss_type == "cross-entropy":
            self.loss_function = nn.CrossEntropyLoss()
        elif loss_type == "bce-with-logit":
            self.loss_function = nn.BCEWithLogitsLoss()
        elif loss_type == "survival_neg_likelihood":
            self.loss_function = SurvivalNegLogLikelihood()
        elif loss_type == "mse":
            self.loss_function = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")
        self.gradient_clip = gradient_clip
        self.device = device
        self.model.to(self.device)

    @staticmethod
    def detach(obj):
        if isinstance(obj, tuple):
            obj = tuple(x.detach() for x in obj)
        elif isinstance(obj, list):
            obj = list(x.detach() for x in obj)
        else:
            obj = obj.detach()
        return obj

    def compute_loss(self, batch):
        features, bag_sizes, targets = (
            batch["features"].to(self.device),
            batch["bag_size"].to(self.device),
            batch["targets"].to(self.device),
        )
        preds = self.model.forward_fn(features, bag_sizes)

        if isinstance(self.loss_function, torch.nn.modules.loss.CrossEntropyLoss):
            preds = preds.view(-1, self.model.head_dim, 1)
            loss = self.loss_function(preds, targets)
        elif isinstance(
            self.loss_function, torch.nn.modules.loss.BCEWithLogitsLoss
        ) or isinstance(self.loss_function, torch.nn.modules.loss.MSELoss):
            targets = targets.float()
            loss = self.loss_function(preds, targets)
        elif isinstance(self.loss_function, SurvivalNegLogLikelihood):
            hazards, survivals, risk_score = preds
            survival_group = targets[:, 0] + 1
            censorship = targets[:, 1]
            loss = self.loss_function(hazards, survival_group, censorship)
            survival_continuous = targets[:, 2]
            targets = (survival_group, censorship, survival_continuous)
        else:
            loss = self.loss_function(preds, targets)

        return preds, targets, loss

    def training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        preds, targets, loss = self.compute_loss(batch)
        loss.backward()
        if self.gradient_clip is not None and self.gradient_clip > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
        self.optimizer.step()
        return self.detach(preds), self.detach(targets), self.detach(loss)

    def validation_step(self, batch, softmax=True, sigmoid=False):
        if softmax and sigmoid:
            raise ValueError(
                f"softmax ({softmax}) and sigmoid ({sigmoid}) can not be used "
                + "together. specify one of them as False"
            )
        self.model.eval()
        preds, targets, loss = self.compute_loss(batch)

        if not isinstance(self.loss_function, SurvivalNegLogLikelihood):
            if softmax:
                preds = nn.functional.softmax(preds, dim=1)
            elif sigmoid:
                preds = nn.functional.sigmoid(preds)

        return (
            self.detach(preds),
            self.detach(targets),
            self.detach(loss),
            batch["sample_ids"],
        )


class BlockLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_blocks: int,
        bias: bool = True,
        flatten: bool = False,
    ):
        super(BlockLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.weight = nn.Parameter(torch.empty((num_blocks, in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(num_blocks, out_features))
        else:
            self.register_parameter("bias", None)
        self.flatten = nn.Flatten(start_dim=0, end_dim=-2) if flatten else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """(c) From: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        :param input: (batch_size, num_blocks, in_features) or (batch_size, num_blocks * in_features)
        :return: res (batch_size, num_blocks, out_features) or (batch_size, num_blocks * out_features) if flatten
        """
        res = torch.matmul(
            input.view(-1, self.num_blocks, 1, self.in_features), self.weight
        ).squeeze(dim=-2)
        res = res + self.bias
        return self.flatten(res)
