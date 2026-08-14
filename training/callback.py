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
        warmup_epoch=0,
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
        self.warmup_epoch = warmup_epoch

    def lr_schedule(self, optimizer, current_epoch_num, lr_init):
        if self.schedule_lr:
            lr = lr_init / 10 ** np.floor(current_epoch_num / 10)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def early_stopping(self, epoch_no):
        early_stop_satisfied = self.stop_cr_counter >= self.patience and self.early_stop
        if early_stop_satisfied and epoch_no >= self.min_epoch_num:
            self.stop = True

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

    @staticmethod
    def write_to_tensoboard(tb_writer, metric, subset, tb_step=0):
        if tb_writer is None:
            return
        for key, val in metric.items():
            if isinstance(val, torch.Tensor):
                val = val.item()
            if val is not None:
                tb_writer.add_scalar(f"{key}/{subset}", val, tb_step)

    def get_best_model(
        self,
        best_model,
        model_state_dict,
        optimizer_state_dict,
        i_epoch,
        loss_val,
        perf_metric_val,
        head_type,
    ):
        # get best model via loss
        better_model_loss = (
            self.stop_criterion == "loss" and loss_val <= best_model["loss_val"]
        )
        # get best model via auc

        better_model_metric = False
        perf_metric = None
        if self.stop_criterion != "loss":
            if head_type == "survival":
                model_val_perf = np.mean([perf_metric_val])
                best_val_perf = np.mean([best_model["perf_metric_val"]])
            else:
                if head_type == "classification":
                    if self.stop_criterion == "perf_metric":
                        perf_metric = "mean"
                    else:
                        perf_metric = self.stop_criterion
                elif head_type == "regression":
                    if self.stop_criterion == "perf_metric":
                        perf_metric = "spearmanr"
                    else:
                        perf_metric = self.stop_criterion

                if perf_metric == "mean":
                    model_val_perf = np.mean(list(perf_metric_val.values()))
                    best_val_perf = np.mean(
                        list(best_model["perf_metric_val"].values())
                    )
                else:
                    model_val_perf = perf_metric_val[perf_metric]
                    best_val_perf = best_model["perf_metric_val"][perf_metric]

            better_model_metric = model_val_perf >= best_val_perf

        # see if model got better in any way
        better_model = better_model_loss or better_model_metric
        if better_model:
            self.stop_cr_counter = 0
            best_model = self.get_model_dict(
                model_state_dict,
                optimizer_state_dict,
                i_epoch,
                loss_val,
                perf_metric_val,
            )
        else:
            self.stop_cr_counter += 1
        return best_model, better_model

    def get_model_dict(
        self, model_state_dict, optimizer_state_dict, i_epoch, loss_val, perf_metric_val
    ):
        model_dict = dict()
        model_dict["loss_val"] = loss_val
        model_dict["perf_metric_val"] = perf_metric_val
        model_dict["epoch"] = i_epoch
        model_dict["model_state_dict"] = copy.deepcopy(model_state_dict)
        model_dict["optimizer_state_dict"] = copy.deepcopy(optimizer_state_dict)
        return model_dict

    def make_checkpoint_backward_compatible(self, checkpoint_state_dict, model):

        # Filter parameters according to use_ppeg
        use_ppeg = getattr(model, "use_ppeg", False)
        model_state_dict = model.state_dict()

        new_state_dict = {}
        for k, v in checkpoint_state_dict.items():
            if "pos_layer" in k:
                if use_ppeg:
                    new_state_dict[k] = v
            else:
                if k in model_state_dict:
                    new_state_dict[k] = v

        return new_state_dict

    def load_checkpoint(self, model, optimizer=None, checkpoint="best"):
        name_load = os.path.join(self.path_checkpoints, f"{checkpoint}_model.pt")
        checkpoint_dict = torch.load(name_load, map_location=self.device)
        checkpoint_state_dict = checkpoint_dict["model_state_dict"]

        model_state_dict = self.make_checkpoint_backward_compatible(
            checkpoint_state_dict, model
        )
        model.load_state_dict(model_state_dict)

        if optimizer is not None and "optimizer_state_dict" in checkpoint_dict:
            optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        return model

    def save_checkpoint_classification(
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
            head_type="classification",
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

    def save_checkpoint_regression(
        self,
        epoch_no,
        loss_this_epoch,
        perf_metrics_this_epoch,
        optimizer_state_dict,
        model_state_dict,
        best_model,
        last_model=False,
        return_args=False,
    ):

        best_model, better_model = self.get_best_model(
            best_model,
            model_state_dict,
            optimizer_state_dict,
            epoch_no,
            loss_this_epoch,
            perf_metrics_this_epoch,
            head_type="regression",
        )
        if better_model:
            name_save = os.path.join(self.path_checkpoints, "best_model.pt")
            torch.save(best_model, name_save)

        if last_model:
            last_model = self.get_model_dict(
                model_state_dict,
                optimizer_state_dict,
                epoch_no,
                loss_this_epoch,
                perf_metrics_this_epoch,
            )
            name_save = os.path.join(self.path_checkpoints, "last_model.pt")
            torch.save(last_model, name_save)

        if return_args:
            return best_model

    def save_checkpoint_survival(
        self,
        epoch_no,
        c_index_all_train,
        c_index_all_val,
        c_index_epoch_train,
        c_index_epoch_val,
        loss_all_train,
        loss_all_val,
        loss_epoch_train,
        loss_epoch_val,
        all_pred_hazard_probs_tr,
        all_pred_hazard_probs_val,
        all_pred_survival_probs_tr,
        all_pred_survival_probs_val,
        all_pred_risks_tr,
        all_pred_risks_val,
        all_true_survivals_tr,
        all_true_survivals_val,
        all_censorships_tr,
        all_censorships_val,
        all_true_survivals_cont_tr,
        all_true_survivals_cont_val,
        optimizer_state_dict,
        model_state_dict,
        best_model,
        last_model=False,
        return_args=False,
    ):

        performance = dict()
        performance["epoch_no"] = epoch_no
        performance["c_index_all_train"] = c_index_all_train
        performance["c_index_all_val"] = c_index_all_val
        performance["loss_all_train"] = loss_all_train
        performance["loss_all_val"] = loss_all_val

        performance["c_index_epoch_train"] = c_index_epoch_train
        performance["c_index_epoch_val"] = c_index_epoch_val
        performance["loss_epoch_train"] = loss_epoch_train
        performance["loss_epoch_val"] = loss_epoch_val

        performance["all_pred_hazard_probs_tr"] = all_pred_hazard_probs_tr
        performance["all_pred_hazard_probs_val"] = all_pred_hazard_probs_val

        performance["all_pred_survival_probs_tr"] = all_pred_survival_probs_tr
        performance["all_pred_survival_probs_val"] = all_pred_survival_probs_val

        performance["all_pred_risks_tr"] = all_pred_risks_tr
        performance["all_pred_risks_val"] = all_pred_risks_val

        performance["all_true_survivals_tr"] = all_true_survivals_tr
        performance["all_true_survivals_val"] = all_true_survivals_val

        performance["all_censorships_tr"] = all_censorships_tr
        performance["all_censorships_val"] = all_censorships_val

        performance["all_true_survivals_cont_tr"] = all_true_survivals_cont_tr
        performance["all_true_survivals_cont_val"] = all_true_survivals_cont_val

        best_model, better_model = self.get_best_model(
            best_model,
            model_state_dict,
            optimizer_state_dict,
            epoch_no,
            loss_epoch_val[-1],
            c_index_epoch_val[-1],
            head_type="survival",
        )
        if better_model:
            torch.save(best_model, os.path.join(self.path_checkpoints, "best_model.pt"))
            torch.save(
                performance, os.path.join(self.path_checkpoints, f"best_performance.pt")
            )

        if last_model:
            last_model = self.get_model_dict(
                model_state_dict,
                optimizer_state_dict,
                epoch_no,
                loss_epoch_val[-1],
                c_index_epoch_val[-1],
            )
            torch.save(last_model, os.path.join(self.path_checkpoints, "last_model.pt"))
            torch.save(
                performance, os.path.join(self.path_checkpoints, f"last_performance.pt")
            )

        if return_args:
            return performance, best_model

    def insert_patch_scores(
        self, all_patch_scores, all_patch_ids, all_preds, predictions
    ):
        if all_patch_scores is not None and len(all_patch_scores) > 0:
            for explanation_type, patch_scores in all_patch_scores.items():
                assert len(all_patch_ids) == len(patch_scores) == len(all_preds)
                predictions.insert(
                    len(predictions.columns),
                    f"patch_scores_{explanation_type}",
                    patch_scores,
                )
        return predictions

    def save_test_results_classification(
        self,
        auc_test,
        loss_test,
        all_preds,
        all_labels,
        label_cols,
        all_sample_ids,
        all_patch_ids=None,
        all_patch_scores=None,
        all_patch_scores_vectors=None,
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
            predictions.insert(len(predictions.columns), "patch_ids", all_patch_ids)
        predictions = self.insert_patch_scores(
            all_patch_scores, all_patch_ids, all_preds, predictions
        )
        predictions = self.insert_patch_scores(
            all_patch_scores_vectors, all_patch_ids, all_preds, predictions
        )

        name_save = os.path.join(self.results_dir, f"test_predictions.csv")
        predictions.to_csv(name_save)

        if return_args:
            return performance, predictions

    def save_test_results_regression(
        self,
        performance,
        loss_test,
        all_preds,
        all_targets,
        all_sample_ids,
        all_patch_ids=None,
        all_patch_scores=None,
        return_args=False,
    ):
        performance["loss_test"] = loss_test
        name_save = os.path.join(self.results_dir, f"test_performance.pt")
        torch.save(performance, name_save)

        predictions = pd.DataFrame(all_sample_ids)
        # expected shape of all_preds (num_samples, 1)
        # expected shape of all_labels (num_samples, 1)
        all_preds, all_targets = all_preds.flatten(), all_targets.flatten()
        assert len(all_preds) == len(all_targets)

        predictions.insert(len(predictions.columns), "prediction", all_preds.tolist())
        predictions.insert(len(predictions.columns), "target", all_targets.tolist())

        if all_patch_scores is not None and len(all_patch_scores) > 0:
            predictions.insert(len(predictions.columns), "patch_ids", all_patch_ids)
        predictions = self.insert_patch_scores(
            all_patch_scores, all_patch_ids, all_preds, predictions
        )

        name_save = os.path.join(self.results_dir, f"test_predictions.csv")
        predictions.to_csv(name_save)

        if return_args:
            return performance, predictions

    def save_test_results_survival(
        self,
        c_index_test,
        loss_test,
        all_pred_hazard_probs,
        all_pred_survival_probs,
        all_pred_risks,
        all_true_survivals,
        all_censorships,
        all_true_survivals_cont,
        all_sample_ids,
        all_patch_ids=None,
        all_patch_scores=None,
        return_args=False,
    ):

        performance = dict()
        performance["c_index_test"] = c_index_test
        performance["loss_test"] = loss_test

        name_save = os.path.join(self.results_dir, f"test_performance.pt")
        torch.save(performance, name_save)

        predictions = pd.DataFrame(all_sample_ids)

        assert (
            len(all_pred_hazard_probs)
            == len(all_pred_survival_probs)
            == len(all_pred_risks)
            == len(all_true_survivals)
            == len(all_censorships)
            == len(all_true_survivals_cont)
        )

        assert all_pred_hazard_probs.shape[1] == all_pred_survival_probs.shape[1]

        for i in range(all_pred_hazard_probs.shape[1]):
            predictions.insert(
                len(predictions.columns),
                f"pred_hazard_prob_{i}",
                all_pred_hazard_probs[:, i].tolist(),
            )

        for i in range(all_pred_survival_probs.shape[1]):
            predictions.insert(
                len(predictions.columns),
                f"pred_survival_prob_{i}",
                all_pred_survival_probs[:, i].tolist(),
            )

        predictions.insert(
            len(predictions.columns), "pred_risk", all_pred_risks.tolist()
        )

        predictions.insert(
            len(predictions.columns), "true_survival_group", all_true_survivals.tolist()
        )
        predictions.insert(
            len(predictions.columns), "censorship", all_censorships.tolist()
        )
        predictions.insert(
            len(predictions.columns),
            "true_survival_cont",
            all_true_survivals_cont.tolist(),
        )

        if all_patch_scores is not None and len(all_patch_scores) > 0:
            for explanation_type, patch_scores in all_patch_scores.items():
                assert len(all_patch_ids) == len(patch_scores) == len(all_pred_risks)
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
