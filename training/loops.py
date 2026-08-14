from collections import defaultdict
import os
import torch
import numpy as np
from tqdm import tqdm
from sksurv.metrics import concordance_index_censored
from training.metrics import RegressionPerformanceMetric


def _get_empty_auc_dict(lbl_names):
    # todo: take this to PerformanceMetrics class in metrics module
    return {f"auc{x.split('label', 1)[1]}": 0 for x in lbl_names}


def _tolist(item):
    if isinstance(item, torch.Tensor) or isinstance(item, np.ndarray):
        item = item.tolist()
    return item


def train_regression_model(
    model,
    classifier,
    optimizer,
    n_epochs,
    lr_init,
    dataloader_train,
    dataloader_val,
    callback,
    ref_value,
    metric_name=None,
    n_epoch_val=1,
    tb_writer=None,
    verbose=False,
):
    metrics_calculator = RegressionPerformanceMetric(metric_name=metric_name)

    n_train_loader = len(dataloader_train)
    n_val_loader = len(dataloader_val)

    # tensorboard
    tb_global_step = 0

    # initialization of the best model
    best_model = callback.get_model_dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        i_epoch=-1,
        loss_val=np.inf,
        perf_metric_val=metrics_calculator.get_empty_metric_dict(),
    )
    # endregion

    for i_epoch in tqdm(range(n_epochs)):

        if verbose:
            print(
                f"epoch #{i_epoch} and stop criterion is {callback.stop_cr_counter} *****"
            )

        callback.lr_schedule(optimizer, i_epoch, lr_init)

        # variables for calculating mean loss and AUC of mini-batches in each epoch
        loss_train, auc_train = 0, 0
        loss_val, auc_val = 0, 0

        all_preds, all_targets = [], []

        # region mini-batch training -------------------
        for i_batch, batch in enumerate(dataloader_train):

            prob_pred_tr, targets_tr, loss = classifier.training_step(batch)

            # process loss values
            if isinstance(loss, dict):
                metrics = loss
                loss = loss.pop("loss")
            else:
                metrics = dict()

            # collecting the loss and AUC info of this mini-batch
            all_preds.append(prob_pred_tr)
            all_targets.append(targets_tr)
            loss_train += (
                loss.item() / n_train_loader
            )  # average of the loss of all mini-batches in this epoch

            if verbose and not i_batch % callback.n_batch_verbose:
                print(f"batch{i_batch} / {n_train_loader} of train")

            callback.write_to_tensoboard(
                tb_writer,
                metric={"loss": loss.item()},
                subset="train",
                tb_step=tb_global_step,
            )
            callback.write_to_tensoboard(
                tb_writer, metric=metrics, subset="train", tb_step=tb_global_step
            )

            if tb_writer is not None:
                tb_global_step += 1

            torch.cuda.empty_cache()

        # collect the epoch training metrics .......
        all_preds = torch.concat(all_preds)
        all_targets = torch.concat(all_targets)

        perf_metrics_tr = metrics_calculator.compute_metrics_regression(
            all_targets, all_preds, ref_value
        )

        if verbose:
            print(
                f"Epoch {i_epoch}: train loss= {loss_train}, train performance={perf_metrics_tr}"
            )

        callback.write_to_tensoboard(
            tb_writer,
            metric={"loss/epoch": loss_train},
            subset="train",
            tb_step=i_epoch,
        )
        callback.write_to_tensoboard(
            tb_writer, metric=perf_metrics_tr, subset="train", tb_step=i_epoch
        )

        # endregion -------------------

        # region mini-batch validation ---------------------------
        model.eval()
        if not i_epoch % n_epoch_val:  # validation for every {n_epoch_val} epochs
            all_preds, all_targets = [], []

            for i_batch, batch in enumerate(dataloader_val):

                prob_pred_val, targets_val, loss, _ = classifier.validation_step(
                    batch, softmax=False
                )

                all_preds.append(prob_pred_val)
                all_targets.append(targets_val)
                loss_val += loss.item() / n_val_loader  # mean val loss of this epoch

                if verbose and not i_batch % callback.n_batch_verbose:
                    print(f"batch{i_batch} / {n_val_loader} of validation")

                torch.cuda.empty_cache()

            # collect the validation metrics ---------------------------
            all_preds = torch.concat(all_preds)
            all_targets = torch.concat(all_targets)
            perf_metrics_val = metrics_calculator.compute_metrics_regression(
                all_targets, all_preds, ref_value
            )

            if verbose:
                print(
                    f"Epoch {i_epoch}: validation loss= {loss_val}, validation performance={perf_metrics_val}"
                )

            callback.write_to_tensoboard(
                tb_writer,
                metric={"loss/epoch": loss_val},
                subset="val",
                tb_step=i_epoch,
            )
            callback.write_to_tensoboard(
                tb_writer, metric=perf_metrics_val, subset="val", tb_step=i_epoch
            )

            # region save checkpoint and check early stopping ---------------------------
            if (
                not i_epoch % callback.checkpoint_epoch
                and i_epoch > callback.warmup_epoch
            ):
                best_model = callback.save_checkpoint_regression(
                    i_epoch,
                    loss_val,
                    perf_metrics_val,
                    optimizer.state_dict(),
                    model.state_dict(),
                    best_model,
                    last_model=False,
                    return_args=True,
                )
                callback.early_stopping(i_epoch)
        if callback.stop:
            break
        # endregion

    # region save checkpoint  ---------------------------
    if n_epochs:
        best_model = callback.save_checkpoint_regression(
            i_epoch,
            loss_val,
            perf_metrics_val,
            optimizer.state_dict(),
            model.state_dict(),
            best_model,
            last_model=True,
            return_args=True,
        )

    # endregion

    return model, best_model


def train_classification_model(
    model,
    classifier,
    optimizer,
    n_epochs,
    lr_init,
    dataloader_train,
    dataloader_val,
    callback,
    label_cols,
    n_epoch_val=1,
    tb_writer=None,
    verbose=False,
):
    n_train_loader = len(dataloader_train)
    n_val_loader = len(dataloader_val)

    # region containers for the results ---------------------------

    # all mini-batch train and val AUC and loss values
    auc_all_train, auc_all_val = [], []
    loss_all_train, loss_all_val = [], []

    # average of the auc and loss in each epoch
    auc_epoch_train, auc_epoch_val = [], []
    loss_epoch_train, loss_epoch_val = [], []

    # tensorboard
    tb_global_step = 0

    # initialization of the best model
    best_model = callback.get_model_dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        i_epoch=-1,
        loss_val=np.inf,
        perf_metric_val=_get_empty_auc_dict(label_cols),
    )
    # endregion

    for i_epoch in tqdm(range(n_epochs)):

        if verbose:
            print(
                f"epoch #{i_epoch} and stop criterion is {callback.stop_cr_counter} *****"
            )

        callback.lr_schedule(optimizer, i_epoch, lr_init)

        # variables for calculating mean loss and AUC of mini-batches in each epoch
        loss_train, auc_train = 0, 0
        loss_val, auc_val = 0, 0

        all_preds, all_labels = [], []

        # region mini-batch training -------------------
        for i_batch, batch in enumerate(dataloader_train):

            prob_pred_tr, labels_tr, loss = classifier.training_step(batch)

            # process loss values
            if isinstance(loss, dict):
                metrics = loss
                loss = loss.pop("loss")
            else:
                metrics = dict()

            # collecting the loss and AUC info of this mini-batch
            all_preds.append(torch.softmax(prob_pred_tr, dim=1))
            all_labels.append(labels_tr)
            loss_train += (
                loss.item() / n_train_loader
            )  # average of the loss of all mini-batches in this epoch
            callback.collect_metric(loss_all_train, loss.item())

            if verbose and not i_batch % callback.n_batch_verbose:
                print(f"batch{i_batch} / {n_train_loader} of train")

            callback.write_to_tensoboard(
                tb_writer,
                metric={"loss": loss.item()},
                subset="train",
                tb_step=tb_global_step,
            )
            callback.write_to_tensoboard(
                tb_writer, metric=metrics, subset="train", tb_step=tb_global_step
            )

            if tb_writer is not None:
                tb_global_step += 1
            torch.cuda.empty_cache()

        # collect the epoch training metrics .......
        all_preds = torch.concat(all_preds)
        all_labels = torch.concat(all_labels)
        auc_train = callback.compute_auc(all_labels, all_preds, lbl_names=label_cols)
        callback.collect_metric(
            loss_epoch_train, loss_train, auc_epoch_train, auc_train
        )
        if verbose:
            print(f"Epoch {i_epoch}: train loss= {loss_train}, train AUC={auc_train}")

        callback.write_to_tensoboard(
            tb_writer,
            metric={"loss/epoch": loss_train},
            subset="train",
            tb_step=i_epoch,
        )
        callback.write_to_tensoboard(
            tb_writer, metric=auc_train, subset="train", tb_step=i_epoch
        )
        # torch.cuda.empty_cache()
        # endregion -------------------

        # region mini-batch validation ---------------------------
        model.eval()
        if not i_epoch % n_epoch_val:  # validation for every {n_epoch_val} epochs
            all_preds, all_labels = [], []

            for i_batch, batch in enumerate(dataloader_val):

                prob_pred_val, labels_val, loss, _ = classifier.validation_step(batch)

                all_preds.append(prob_pred_val)
                all_labels.append(labels_val)
                loss_val += loss.item() / n_val_loader  # mean val loss of this epoch
                callback.collect_metric(loss_all_val, loss.item())

                if verbose and not i_batch % callback.n_batch_verbose:
                    print(f"batch{i_batch} / {n_val_loader} of validation")

                torch.cuda.empty_cache()

            # collect the validation metrics ---------------------------
            all_preds = torch.concat(all_preds)
            all_labels = torch.concat(all_labels)
            auc_val = callback.compute_auc(all_labels, all_preds, lbl_names=label_cols)
            callback.collect_metric(loss_epoch_val, loss_val, auc_epoch_val, auc_val)
            if verbose:
                print(
                    f"Epoch {i_epoch}: validation loss= {loss_val}, validation AUC={auc_val}"
                )

            callback.write_to_tensoboard(
                tb_writer,
                metric={"loss/epoch": loss_val},
                subset="val",
                tb_step=i_epoch,
            )
            callback.write_to_tensoboard(
                tb_writer, metric=auc_val, subset="val", tb_step=i_epoch
            )
        # endregion

        # region save checkpoint and check early stopping ---------------------------
        if not i_epoch % callback.checkpoint_epoch and i_epoch > callback.warmup_epoch:
            _, best_model = callback.save_checkpoint_classification(
                i_epoch,
                auc_all_train,
                auc_all_val,
                loss_all_train,
                loss_all_val,
                auc_epoch_train,
                auc_epoch_val,
                loss_epoch_train,
                loss_epoch_val,
                optimizer.state_dict(),
                model.state_dict(),
                best_model,
                return_args=True,
            )
            callback.early_stopping(i_epoch)
        if callback.stop:
            break
        # endregion

    # region save checkpoint  ---------------------------
    performance = None
    if n_epochs:
        performance, best_model = callback.save_checkpoint_classification(
            n_epochs - 1,
            auc_all_train,
            auc_all_val,
            loss_all_train,
            loss_all_val,
            auc_epoch_train,
            auc_epoch_val,
            loss_epoch_train,
            loss_epoch_val,
            optimizer.state_dict(),
            model.state_dict(),
            best_model,
            last_model=True,
            return_args=True,
        )

    # endregion

    return performance, model, best_model


def train_survival_model(
    model,
    classifier,
    optimizer,
    n_epochs,
    lr_init,
    dataloader_train,
    dataloader_val,
    callback,
    n_epoch_val=1,
    tb_writer=None,
    verbose=False,
):
    n_train_loader = len(dataloader_train)
    n_val_loader = len(dataloader_val)

    # region containers for the results ---------------------------

    # all mini-batch train and val C-Index and loss values
    loss_all_train, loss_all_val = [], []
    c_index_all_train, c_index_all_val = [], []

    # average of the C-Index and loss in each epoch
    loss_epoch_train, loss_epoch_val = [], []
    c_index_epoch_train, c_index_epoch_val = [], []

    # tensorboard
    tb_global_step = 0

    # initialization of the best model
    best_model = callback.get_model_dict(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        i_epoch=-1,
        loss_val=np.inf,
        perf_metric_val=0,
    )

    performance = None
    # endregion

    for i_epoch in tqdm(range(n_epochs)):

        if verbose:
            print(
                f"epoch #{i_epoch} and stop criterion is {callback.stop_cr_counter} *****"
            )

        callback.lr_schedule(optimizer, i_epoch, lr_init)

        # variables for calculating mean loss and C-Index of mini-batches in each epoch
        loss_train, loss_val, total_loss = 0, 0, 0
        all_pred_hazard_probs_tr, all_pred_survival_probs_tr, all_pred_risks_tr = (
            [],
            [],
            [],
        )
        all_true_survivals_tr, all_censorships_tr, all_true_survivals_cont_tr = (
            [],
            [],
            [],
        )
        """
        all_pred_hazard_probs_tr: predicted hazard probability of all samples
        all_pred_survival_probs_tr: predicted survival probability of all samples
        all_pred_risks_tr: predicted risk scores of all samples
        all_true_survivals_tr: true survival groups of all samples
        all_true_survivals_cont_tr: true continuous survival value
        all_censorships_tr: the censorship status of all samples 
        """

        # region mini-batch training -------------------
        for i_batch, batch in enumerate(dataloader_train):

            preds, targets, loss = classifier.training_step(batch)
            pred_hazard_probs, pred_survival_probs, pred_risk_scores = preds
            true_survival_group, censorship_status, true_survival_continuous = targets

            # process loss values
            if isinstance(loss, dict):
                metrics = loss
                loss = loss.pop("loss")
            else:
                metrics = dict()

            # collecting the info of this batch
            all_pred_hazard_probs_tr.append(pred_hazard_probs)
            all_pred_survival_probs_tr.append(pred_survival_probs)
            all_pred_risks_tr.append(pred_risk_scores)

            all_true_survivals_tr.append(true_survival_group)
            all_censorships_tr.append(censorship_status)
            all_true_survivals_cont_tr.append(true_survival_continuous)

            loss_train += (
                loss.item() / n_train_loader
            )  # average of the loss of all mini-batches in this epoch
            callback.collect_metric(loss_all_train, loss.item())
            total_loss += loss.item()

            if verbose and not i_batch % callback.n_batch_verbose:
                print(f"batch{i_batch} / {n_train_loader} of train")

            callback.write_to_tensoboard(
                tb_writer,
                metric={"loss": loss.item()},
                subset="train",
                tb_step=tb_global_step,
            )
            callback.write_to_tensoboard(
                tb_writer, metric=metrics, subset="train", tb_step=tb_global_step
            )

            if tb_writer is not None:
                tb_global_step += 1
            torch.cuda.empty_cache()

        # collect the epoch training metrics .......
        all_pred_hazard_probs_tr = torch.concat(all_pred_hazard_probs_tr).cpu().numpy()
        all_pred_survival_probs_tr = (
            torch.concat(all_pred_survival_probs_tr).cpu().numpy()
        )
        all_pred_risks_tr = torch.concat(all_pred_risks_tr).cpu().numpy()

        all_true_survivals_tr = torch.concat(all_true_survivals_tr).cpu().numpy()
        all_censorships_tr = torch.concat(all_censorships_tr).cpu().numpy()
        all_true_survivals_cont_tr = (
            torch.concat(all_true_survivals_cont_tr).cpu().numpy()
        )

        c_index_train = concordance_index_censored(
            all_censorships_tr == 0,
            all_true_survivals_tr,
            all_pred_risks_tr,
            tied_tol=1e-8,
        )[0]
        callback.collect_metric(
            loss_epoch_train, loss_train, c_index_epoch_train, c_index_train
        )

        callback.write_to_tensoboard(
            tb_writer,
            metric={"loss/epoch": total_loss},
            subset="train",
            tb_step=i_epoch,
        )
        callback.write_to_tensoboard(
            tb_writer,
            metric={"C-Index": c_index_train},
            subset="train",
            tb_step=i_epoch,
        )

        if verbose:
            print(f"Epoch {i_epoch}: train loss= {loss_train}")

        torch.cuda.empty_cache()
        # endregion -------------------

        # region mini-batch validation ---------------------------
        model.eval()
        all_pred_hazard_probs_val, all_pred_survival_probs_val, all_pred_risks_val = (
            [],
            [],
            [],
        )
        all_true_survivals_val, all_censorships_val, all_true_survivals_cont_val = (
            [],
            [],
            [],
        )
        if not i_epoch % n_epoch_val:  # validation for every {n_epoch_val} epochs
            for i_batch, batch in enumerate(dataloader_val):
                preds, targets, loss, _ = classifier.validation_step(batch)
                pred_hazard_probs, pred_survival_probs, pred_risk_scores = preds
                true_survival_group, censorship_status, true_survival_continuous = (
                    targets
                )

                all_pred_hazard_probs_val.append(pred_hazard_probs)
                all_pred_survival_probs_val.append(pred_survival_probs)
                all_pred_risks_val.append(pred_risk_scores)

                all_true_survivals_val.append(true_survival_group)
                all_censorships_val.append(censorship_status)
                all_true_survivals_cont_val.append(true_survival_continuous)

                loss_val += loss.item() / n_val_loader  # mean val loss of this epoch
                callback.collect_metric(loss_all_val, loss.item())

                if verbose and not i_batch % callback.n_batch_verbose:
                    print(f"batch{i_batch} / {n_val_loader} of validation")

                torch.cuda.empty_cache()

            # collect the validation metrics ---------------------------
            all_pred_hazard_probs_val = (
                torch.concat(all_pred_hazard_probs_val).cpu().numpy()
            )
            all_pred_survival_probs_val = (
                torch.concat(all_pred_survival_probs_val).cpu().numpy()
            )
            all_pred_risks_val = torch.concat(all_pred_risks_val).cpu().numpy()

            all_true_survivals_val = torch.concat(all_true_survivals_val).cpu().numpy()
            all_censorships_val = torch.concat(all_censorships_val).cpu().numpy()
            all_true_survivals_cont_val = (
                torch.concat(all_true_survivals_cont_val).cpu().numpy()
            )

            c_index_val = concordance_index_censored(
                all_censorships_val == 0,
                all_true_survivals_val,
                all_pred_risks_val,
                tied_tol=1e-8,
            )[0]
            callback.collect_metric(
                loss_epoch_val, loss_val, c_index_epoch_val, c_index_val
            )

            if verbose:
                print(f"Epoch {i_epoch}: validation loss= {loss_val}")

            callback.write_to_tensoboard(
                tb_writer,
                metric={"loss/epoch": loss_val},
                subset="val",
                tb_step=i_epoch,
            )
            callback.write_to_tensoboard(
                tb_writer,
                metric={"C-Index": c_index_val},
                subset="val",
                tb_step=i_epoch,
            )

        # endregion

        # region save checkpoint and check early stopping ---------------------------
        if (
            not i_epoch % callback.checkpoint_epoch or i_epoch == n_epochs - 1
        ) and i_epoch > callback.warmup_epoch:
            performance, best_model = callback.save_checkpoint_survival(
                i_epoch,
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
                all_true_survivals_tr - 1,
                all_true_survivals_val - 1,
                all_censorships_tr,
                all_censorships_val,
                all_true_survivals_cont_tr,
                all_true_survivals_cont_val,
                optimizer.state_dict(),
                model.state_dict(),
                best_model,
                last_model=(i_epoch == n_epochs - 1),
                return_args=True,
            )
            callback.early_stopping(i_epoch)
        if callback.stop:
            break
        # endregion

    return performance, model, best_model


def test_classification_model(
    model,
    classifier,
    dataloader_test,
    callback,
    label_cols,
    xmodel=None,
    explanation_types=None,
    save_explanation_vectors=False,
    compute_auc=True,
    tb_writer=None,
    verbose=False,
):
    model.eval()

    (
        all_preds,
        all_labels,
        loss_test,
        all_sample_ids,
        all_patch_scores,
        all_patch_ids,
    ) = ([], [], [], {}, defaultdict(list), [])
    all_patch_scores_vectors = defaultdict(list)

    for i_batch, batch in enumerate(tqdm(dataloader_test)):

        prob_pred_val, labels_val, loss, sample_ids = classifier.validation_step(
            batch, softmax=True
        )

        all_preds.append(prob_pred_val)
        all_labels.append(labels_val)
        loss_test.append(loss.item())
        if isinstance(sample_ids, dict):
            all_sample_ids = {
                key: all_sample_ids.get(key, []) + _tolist(ids)
                for key, ids in sample_ids.items()
            }
        else:
            all_sample_ids = {
                "sample_id": all_sample_ids.get("sample_id", []) + _tolist(sample_ids)
            }
        torch.cuda.empty_cache()

        batch_slide_id = batch["sample_ids"]["slide_id"][0]
        if xmodel is not None:
            for explanation_type in explanation_types:
                patch_scores, patch_scores_vectors = xmodel.get_heatmap(
                    batch, explanation_type, verbose
                )
                all_patch_scores[explanation_type].append(patch_scores.tolist())
                if patch_scores_vectors is not None and save_explanation_vectors:
                    vectors_dir = os.path.join(
                        callback.results_dir, f"{explanation_type}_vectors"
                    )
                    if not os.path.isdir(vectors_dir):
                        os.makedirs(vectors_dir)
                    np.save(
                        os.path.join(vectors_dir, f"{batch_slide_id}.npy"),
                        patch_scores_vectors,
                    )

            all_patch_ids.append(batch["patch_ids"].tolist())

        torch.cuda.empty_cache()

    all_preds = torch.concat(all_preds)
    all_labels = torch.concat(all_labels)
    loss_test = torch.tensor(loss_test).mean(dim=0).item()
    auc_test = (
        callback.compute_auc(all_labels, all_preds, lbl_names=label_cols)
        if compute_auc
        else None
    )

    if verbose:
        print(f"Test loss={loss_test}, test AUC={auc_test}")

    callback.write_to_tensoboard(
        tb_writer, metric={"loss": loss_test}, subset="test", tb_step=0
    )
    callback.write_to_tensoboard(tb_writer, metric=auc_test, subset="test", tb_step=0)

    results = callback.save_test_results_classification(
        auc_test,
        loss_test,
        all_preds,
        all_labels,
        label_cols,
        all_sample_ids,
        all_patch_ids,
        all_patch_scores,
        all_patch_scores_vectors,
        return_args=False,
    )
    return results


def test_regression_model(
    model,
    classifier,
    dataloader_test,
    callback,
    xmodel=None,
    explanation_types=None,
    save_explanation_vectors=False,
    ref_value=None,
    tb_writer=None,
    verbose=False,
):
    model.eval()
    metrics_calculator = RegressionPerformanceMetric(metric_name=None)
    loss_test = []
    all_targets, all_preds = [], []
    all_sample_ids = {}
    all_patch_scores, all_patch_ids = defaultdict(list), []

    for i_batch, batch in enumerate(tqdm(dataloader_test)):

        prob_pred_val, targets_val, loss, sample_ids = classifier.validation_step(
            batch, softmax=False
        )

        all_preds.append(prob_pred_val)
        all_targets.append(targets_val)
        loss_test.append(loss.item())
        if isinstance(sample_ids, dict):
            all_sample_ids = {
                key: all_sample_ids.get(key, []) + _tolist(ids)
                for key, ids in sample_ids.items()
            }
        else:
            all_sample_ids = {
                "sample_id": all_sample_ids.get("sample_id", []) + _tolist(sample_ids)
            }
        torch.cuda.empty_cache()

        batch_slide_id = batch["sample_ids"]["slide_id"][0]
        if xmodel is not None:
            for explanation_type in explanation_types:
                patch_scores, patch_scores_vectors = xmodel.get_heatmap(
                    batch, explanation_type, verbose
                )
                all_patch_scores[explanation_type].append(patch_scores.tolist())
                if patch_scores_vectors is not None and save_explanation_vectors:
                    vectors_dir = os.path.join(
                        callback.results_dir, f"{explanation_type}_vectors"
                    )
                    if not os.path.isdir(vectors_dir):
                        os.makedirs(vectors_dir)
                    np.save(
                        os.path.join(vectors_dir, f"{batch_slide_id}.npy"),
                        patch_scores_vectors,
                    )

            all_patch_ids.append(batch["patch_ids"].tolist())

        torch.cuda.empty_cache()

    all_preds = torch.concat(all_preds)
    all_targets = torch.concat(all_targets)
    loss_test = torch.tensor(loss_test).mean(dim=0).item()
    perf_metrics = metrics_calculator.compute_metrics_regression(
        all_targets, all_preds, ref_value
    )

    callback.write_to_tensoboard(
        tb_writer, metric={"loss": loss_test}, subset="test", tb_step=0
    )
    callback.write_to_tensoboard(
        tb_writer, metric=perf_metrics, subset="test", tb_step=0
    )

    results = callback.save_test_results_regression(
        perf_metrics,
        loss_test,
        all_preds,
        all_targets,
        all_sample_ids,
        all_patch_ids,
        all_patch_scores,
        return_args=False,
    )

    return results


def test_survival_model(
    model,
    classifier,
    dataloader_test,
    callback,
    xmodel=None,
    explanation_types=None,
    save_explanation_vectors=False,
    tb_writer=None,
    verbose=False,
):
    model.eval()

    all_pred_hazard_probs, all_pred_survival_probs, all_pred_risks = [], [], []
    all_true_survivals, all_censorships, all_true_survivals_cont = [], [], []
    """
    all_pred_hazard_probs: predicted hazard probability of all samples
    all_pred_survival_probs: predicted survival probability of all samples
    all_pred_risks: predicted risk scores of all samples
    all_true_survivals: true survival groups of all samples
    all_censorships: the censorship status of all samples 
    """
    loss_test = []
    all_sample_ids = {}
    all_patch_scores, all_patch_ids = defaultdict(list), []

    for i_batch, batch in enumerate(tqdm(dataloader_test)):

        preds, targets, loss, sample_ids = classifier.validation_step(batch)
        pred_hazard_probs, pred_survival_probs, pred_risk_scores = preds
        true_survival_group, censorship_status, true_survival_continuous = targets

        all_pred_hazard_probs.append(pred_hazard_probs)
        all_pred_survival_probs.append(pred_survival_probs)
        all_pred_risks.append(pred_risk_scores)

        all_true_survivals.append(true_survival_group)
        all_censorships.append(censorship_status)
        all_true_survivals_cont.append(true_survival_continuous)

        loss_test.append(loss.item())

        if isinstance(sample_ids, dict):
            all_sample_ids = {
                key: all_sample_ids.get(key, []) + _tolist(ids)
                for key, ids in sample_ids.items()
            }
        else:
            all_sample_ids = {
                "sample_id": all_sample_ids.get("sample_id", []) + _tolist(sample_ids)
            }
        torch.cuda.empty_cache()

        batch_slide_id = batch["sample_ids"]["slide_id"][0]
        if xmodel is not None:
            for explanation_type in explanation_types:
                patch_scores, patch_scores_vectors = xmodel.get_heatmap(
                    batch, explanation_type, verbose
                )
                all_patch_scores[explanation_type].append(patch_scores.tolist())
                if patch_scores_vectors is not None and save_explanation_vectors:
                    vectors_dir = os.path.join(
                        callback.results_dir, f"{explanation_type}_vectors"
                    )
                    if not os.path.isdir(vectors_dir):
                        os.makedirs(vectors_dir)
                    np.save(
                        os.path.join(vectors_dir, f"{batch_slide_id}.npy"),
                        patch_scores_vectors,
                    )

            all_patch_ids.append(batch["patch_ids"].tolist())
        torch.cuda.empty_cache()

    loss_test = torch.tensor(loss_test).mean(dim=0).item()

    all_pred_hazard_probs = torch.concat(all_pred_hazard_probs).cpu().numpy()
    all_pred_survival_probs = torch.concat(all_pred_survival_probs).cpu().numpy()
    all_pred_risks = torch.concat(all_pred_risks).cpu().numpy()

    all_true_survivals = torch.concat(all_true_survivals).cpu().numpy()
    all_censorships = torch.concat(all_censorships).cpu().numpy()
    all_true_survivals_cont = torch.concat(all_true_survivals_cont).cpu().numpy()

    c_index_test = concordance_index_censored(
        all_censorships == 0, all_true_survivals, all_pred_risks, tied_tol=1e-8
    )[0]

    if verbose:
        print(f"Test loss={loss_test}, test C-index={c_index_test}")

    callback.write_to_tensoboard(
        tb_writer, metric={"loss": loss_test}, subset="test", tb_step=0
    )
    callback.write_to_tensoboard(
        tb_writer, metric={"C-Index": c_index_test}, subset="test", tb_step=0
    )

    results = callback.save_test_results_survival(
        c_index_test,
        loss_test,
        all_pred_hazard_probs,
        all_pred_survival_probs,
        all_pred_risks,
        all_true_survivals - 1,
        all_censorships,
        all_true_survivals_cont,
        all_sample_ids,
        all_patch_ids=all_patch_ids,
        all_patch_scores=all_patch_scores,
        return_args=True,
    )
    return results
