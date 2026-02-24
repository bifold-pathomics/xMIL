import torch
import torch.nn as nn


class Classifier:

    def __init__(
        self,
        model,
        learning_rate,
        weight_decay,
        optimizer="SGD",
        objective="cross-entropy",
        gradient_clip=None,
        device=torch.device("cpu"),
    ):
        self.model = model

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

        # Set up criterion
        if objective == "cross-entropy":
            self.criterion = torch.nn.CrossEntropyLoss()
        elif objective == "bce-with-logit":
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown objective: {objective}")
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

        if isinstance(self.criterion, torch.nn.modules.loss.CrossEntropyLoss):
            preds = preds.view(-1, self.model.n_classes, 1)
            loss = self.criterion(preds, targets)
        elif isinstance(self.criterion, torch.nn.modules.loss.BCEWithLogitsLoss):
            targets = targets.float()
            loss = self.criterion(preds, targets)
        else:
            loss = self.criterion(preds, targets)

        return preds, targets, loss

    def training_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        preds, targets, loss = self.compute_loss(batch)
        loss.backward()
        if self.gradient_clip is not None:
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
