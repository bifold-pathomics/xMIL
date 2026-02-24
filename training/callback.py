import os
import copy

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


class Callback:

    def __init__(
        self,
        schedule_lr,
        checkpoint_epoch,
        path_checkpoints,
        n_batch_verbose=50,
        stop_criterion="loss",
        patience=3,
        min_epoch_num=10,
        early_stop=False,
        device="cpu",
        results_dir=None,
    ):
        self.schedule_lr = schedule_lr
        self.checkpoint_epoch = checkpoint_epoch
        self.path_checkpoints = path_checkpoints
        self.n_batch_verbose = n_batch_verbose
        self.min_epoch_num = min_epoch_num
        self.early_stop = early_stop
        self.stop_criterion = stop_criterion
        self.patience = patience
        self.stop_cr_counter = 0
        self.stop = False
        self.device = device
        self.results_dir = results_dir if results_dir is not None else path_checkpoints

    def lr_schedule(self, optimizer, current_epoch_num, lr_init):
        if self.schedule_lr:
            lr = lr_init / 10 ** np.floor(current_epoch_num / 10)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    @staticmethod
    def compute_auc(lbl, prob_model, lbl_names=None):
        lbl = lbl.cpu().numpy()
        prob_model = (
            prob_model[:, -1, :].detach().cpu().numpy()
        )  # probability of class 1
        if lbl_names is None:
            auc_targets = [f"auc_target_{i}" for i in range(lbl.shape[-1])]
        else:
            assert len(lbl_names) == lbl.shape[-1]
            auc_targets = [f"auc{x.split('label', 1)[1]}" for x in lbl_names]
        aucs = {}
        for i, target in zip(range(lbl.shape[-1]), auc_targets):
            curr_lbl = lbl[:, i]
            curr_pred = prob_model[:, i]
            aucs[target] = roc_auc_score(curr_lbl, curr_pred)
        return aucs

    @staticmethod
    def collect_metric(*args):
        i = 0
        while i < len(args) - 1:
            args[i].append(args[i + 1])
            i += 2

    def save_checkpoint(
        self,
        epoch_no,
        auc_all_train,
        auc_all_val,
        loss_all_train,
        loss_all_val,
        auc_epoch_train,
        auc_epoch_val,
        loss_epoch_train,
        loss_epoch_val,
        optimizer_state_dict,
        model_state_dict,
        best_model,
        last_model=False,
        return_args=False,
    ):
        performance = dict()
        performance["auc_all_train"] = auc_all_train
        performance["auc_all_val"] = auc_all_val
        performance["loss_all_train"] = loss_all_train
        performance["loss_all_val"] = loss_all_val

        performance["auc_epoch_train"] = auc_epoch_train
        performance["auc_epoch_val"] = auc_epoch_val
        performance["loss_epoch_train"] = loss_epoch_train
        performance["loss_epoch_val"] = loss_epoch_val
        performance["epoch_no"] = epoch_no

        best_model, better_model = self.get_best_model(
            best_model,
            model_state_dict,
            optimizer_state_dict,
            epoch_no,
            loss_epoch_val[-1],
            auc_epoch_val[-1],
        )
        if better_model:
            name_save = os.path.join(self.path_checkpoints, "best_model.pt")
            torch.save(best_model, name_save)
            name_save = os.path.join(self.path_checkpoints, f"best_performance.pt")
            torch.save(performance, name_save)

        if last_model:
            last_model = self.get_model_dict(
                model_state_dict,
                optimizer_state_dict,
                epoch_no,
                loss_epoch_val[-1],
                auc_epoch_val[-1],
            )
            name_save = os.path.join(self.path_checkpoints, "last_model.pt")
            torch.save(last_model, name_save)
            name_save = os.path.join(self.path_checkpoints, f"last_performance.pt")
            torch.save(performance, name_save)

        if return_args:
            return performance, best_model

    def get_best_model(
        self,
        best_model,
        model_state_dict,
        optimizer_state_dict,
        i_epoch,
        loss_val,
        auc_val,
    ):
        # get best model via loss
        better_model_loss = (
            self.stop_criterion == "loss" and loss_val <= best_model["loss_val"]
        )
        # get best model via auc
        model_mean_target_auc = np.mean(list(auc_val.values()))
        best_model_mean_target_auc = np.mean(list(best_model["auc_val"].values()))
        better_model_auc = (
            self.stop_criterion == "auc"
            and model_mean_target_auc >= best_model_mean_target_auc
        )
        # see if model got better in any way
        better_model = better_model_loss or better_model_auc
        if better_model:
            self.stop_cr_counter = 0
            best_model = self.get_model_dict(
                model_state_dict, optimizer_state_dict, i_epoch, loss_val, auc_val
            )
        else:
            self.stop_cr_counter += 1
        return best_model, better_model

    def get_model_dict(
        self, model_state_dict, optimizer_state_dict, i_epoch, loss_val, auc_val
    ):
        model_dict = dict()
        model_dict["loss_val"] = loss_val
        model_dict["auc_val"] = auc_val
        model_dict["epoch"] = i_epoch
        model_dict["model_state_dict"] = copy.deepcopy(model_state_dict)
        model_dict["optimizer_state_dict"] = copy.deepcopy(optimizer_state_dict)
        return model_dict

    def early_stopping(self, epoch_no):
        early_stop_satisfied = self.stop_cr_counter >= self.patience and self.early_stop
        if early_stop_satisfied and epoch_no >= self.min_epoch_num:
            self.stop = True

    def load_checkpoint(self, model, optimizer=None, checkpoint="best"):
        name_load = os.path.join(self.path_checkpoints, f"{checkpoint}_model.pt")
        model_dict = torch.load(name_load, map_location=self.device)
        model.load_state_dict(model_dict["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(model_dict["optimizer_state_dict"])
        return model

    def save_test_results(
        self,
        auc_test,
        loss_test,
        all_preds,
        all_labels,
        label_cols,
        all_sample_ids,
        all_patch_ids=None,
        all_patch_scores=None,
        return_args=False,
    ):
        performance = dict()
        performance["auc_test"] = auc_test
        performance["loss_test"] = loss_test
        name_save = os.path.join(self.results_dir, f"test_performance.pt")
        torch.save(performance, name_save)

        predictions = pd.DataFrame(all_sample_ids)
        # expected shape of all_preds (num_samples, num_classes, num_targets)
        # expected shape of all_labels (num_samples, num_targets)
        assert all_preds.shape[2] == all_labels.shape[1]
        if len(label_cols) == 0:
            label_cols = [f"label_{i}" for i in range(all_labels.shape[1])]
            pred_cols = [f"prediction_score_{i}" for i in range(all_labels.shape[1])]
        else:
            pred_cols = [
                "prediction_score" + lbl.split("label", 1)[1] for lbl in label_cols
            ]
        if len(all_preds.shape) > 1:
            all_preds = all_preds[:, -1, :]  # get prediction for class 1
        for i, col in enumerate(pred_cols):
            predictions.insert(len(predictions.columns), col, all_preds[:, i].tolist())
        for i, col in enumerate(label_cols):
            predictions.insert(len(predictions.columns), col, all_labels[:, i].tolist())

        if all_patch_scores is not None and len(all_patch_scores) > 0:
            for explanation_type, patch_scores in all_patch_scores.items():
                assert len(all_patch_ids) == len(patch_scores) == len(all_preds)
                predictions.insert(
                    len(predictions.columns),
                    f"patch_scores_{explanation_type}",
                    patch_scores,
                )
            predictions.insert(len(predictions.columns), "patch_ids", all_patch_ids)

        name_save = os.path.join(self.results_dir, f"test_predictions.csv")
        predictions.to_csv(name_save)

        if return_args:
            return performance, predictions
