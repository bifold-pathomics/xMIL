from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm


def _get_empty_auc_dict(lbl_names):
    return {f"auc{x.split('label', 1)[1]}": 0 for x in lbl_names}


def train_classification_model(
        model, classifier, optimizer, n_epochs, lr_init, dataloader_train, dataloader_val, callback, label_cols,
        n_epoch_val=1, tb_writer=None, verbose=False):
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
    best_model = callback.get_model_dict(model_state_dict=model.state_dict(),
                                         optimizer_state_dict=optimizer.state_dict(),
                                         i_epoch=-1,
                                         loss_val=np.inf,
                                         auc_val=_get_empty_auc_dict(label_cols))
    # endregion

    for i_epoch in tqdm(range(n_epochs)):

        if verbose:
            print(f'epoch #{i_epoch} and stop criterion is {callback.stop_cr_counter} *****')

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
                loss = loss.pop('loss')
            else:
                metrics = dict()

            # collecting the loss and AUC info of this mini-batch
            all_preds.append(torch.softmax(prob_pred_tr, dim=1))
            all_labels.append(labels_tr)
            loss_train += (loss.item() / n_train_loader)  # average of the loss of all mini-batches in this epoch
            callback.collect_metric(loss_all_train, loss.item())

            if verbose and not i_batch % callback.n_batch_verbose:
                print(f'batch{i_batch} / {n_train_loader} of train')

            if tb_writer is not None:
                tb_writer.add_scalar('loss/train', loss.item(), tb_global_step)
                for key, val in metrics.items():
                    tb_writer.add_scalar(f'{key}/train', val.item(), tb_global_step)
                tb_global_step += 1
            torch.cuda.empty_cache()

        # collect the epoch training metrics .......
        all_preds = torch.concat(all_preds)
        all_labels = torch.concat(all_labels)
        auc_train = callback.compute_auc(all_labels, all_preds, lbl_names=label_cols)
        callback.collect_metric(loss_epoch_train, loss_train, auc_epoch_train, auc_train)
        if verbose:
            print(f'Epoch {i_epoch}: train loss= {loss_train}, train AUC={auc_train}')

        if tb_writer is not None:
            for key, val in auc_train.items():
                tb_writer.add_scalar(f'{key}/train', val, tb_global_step)

        # torch.cuda.empty_cache()
        # endregion -------------------

        # region mini-batch validation ---------------------------
        model.eval()
        if not i_epoch % n_epoch_val:  # validation for every {n_epoch_val} epochs
            all_preds, all_labels = [], []

            for i_batch, batch in enumerate(dataloader_val):

                prob_pred_val, labels_val, loss, _ = classifier.validation_step(batch, softmax=True)

                all_preds.append(prob_pred_val)
                all_labels.append(labels_val)
                loss_val += loss.item() / n_val_loader  # mean val loss of this epoch
                callback.collect_metric(loss_all_val, loss.item())

                if verbose and not i_batch % callback.n_batch_verbose:
                    print(f'batch{i_batch} / {n_val_loader} of validation')

                torch.cuda.empty_cache()

            # collect the validation metrics ---------------------------
            all_preds = torch.concat(all_preds)
            all_labels = torch.concat(all_labels)
            auc_val = callback.compute_auc(all_labels, all_preds, lbl_names=label_cols)
            callback.collect_metric(loss_epoch_val, loss_val, auc_epoch_val, auc_val)
            if verbose:
                print(f'Epoch {i_epoch}: validation loss= {loss_val}, validation AUC={auc_val}')

            if tb_writer is not None:
                tb_writer.add_scalar('loss/val', loss_val, tb_global_step)
                for key, val in auc_val.items():
                    tb_writer.add_scalar(f'{key}/val', val, tb_global_step)
        # endregion

        # region save checkpoint and check early stopping ---------------------------
        if not i_epoch % callback.checkpoint_epoch:
            _, best_model = callback.save_checkpoint(
                i_epoch, auc_all_train, auc_all_val, loss_all_train, loss_all_val,
                auc_epoch_train, auc_epoch_val, loss_epoch_train, loss_epoch_val,
                optimizer.state_dict(), model.state_dict(), best_model,
                return_args=True)
            callback.early_stopping(i_epoch)
        if callback.stop:
            break
        # endregion

    # region save checkpoint  ---------------------------
    performance = None
    if n_epochs:
        performance, best_model = callback.save_checkpoint(
            n_epochs - 1, auc_all_train, auc_all_val, loss_all_train, loss_all_val,
            auc_epoch_train, auc_epoch_val, loss_epoch_train, loss_epoch_val,
            optimizer.state_dict(), model.state_dict(), best_model, last_model=True,
            return_args=True)

    # endregion

    return performance, model, best_model


def test_classification_model(
        model, classifier, dataloader_test, callback, label_cols, xmodel=None, explanation_types=None, tb_writer=None,
        verbose=False):
    model.eval()

    all_preds, all_labels, loss_test, all_sample_ids, \
        all_patch_scores, all_patch_ids = [], [], [], {}, defaultdict(list), []

    for i_batch, batch in enumerate(tqdm(dataloader_test)):

        prob_pred_val, labels_val, loss, pred_metadata = classifier.validation_step(batch, softmax=True)

        all_preds.append(prob_pred_val)
        all_labels.append(labels_val)
        loss_test.append(loss.item())
        if isinstance(pred_metadata['slide_id'], dict):
            all_sample_ids = {key: all_sample_ids.get(key, []) + ids for key, ids in pred_metadata['slide_id'].items()}
        else:
            all_sample_ids = {'slide_id': all_sample_ids.get('slide_id', []) + pred_metadata['slide_id']}

        torch.cuda.empty_cache()

        if xmodel is not None:
            for explanation_type in explanation_types:
                patch_scores = xmodel.get_heatmap(batch, explanation_type, verbose)
                all_patch_scores[explanation_type].append(patch_scores.tolist())

            all_patch_ids.append(batch['patch_ids'].tolist())

        torch.cuda.empty_cache()

    all_preds = torch.concat(all_preds)
    all_labels = torch.concat(all_labels)
    loss_test = torch.tensor(loss_test).mean(dim=0).item()
    auc_test = callback.compute_auc(all_labels, all_preds, lbl_names=label_cols)

    if verbose:
        print(f'Test loss={loss_test}, test AUC={auc_test}')

    if tb_writer is not None:
        tb_writer.add_scalar('loss/test', loss_test, 0)
        for key, val in auc_test.items():
            tb_writer.add_scalar(f'{key}/test', val, 0)

    results = callback.save_test_results(auc_test, loss_test, all_preds, all_labels, label_cols,
                                         all_sample_ids, all_patch_ids, all_patch_scores, return_args=False)

    return results
