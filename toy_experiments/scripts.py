import copy

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


def train_model(classifier, num_classes, data_loader_train, data_loader_val, batch_size,
                num_epochs=100, warmup=15, tolerance=0.0, patience=5, checkpoint='best'):

    min_val_loss, stop_count = 1e9, 0
    checkpoint_best, metrics_best = None, None

    for epoch in range(num_epochs):
        train_preds, train_labels, train_loss, val_preds, val_labels, val_loss = [], [], 0, [], [], 0
        for batch in data_loader_train:
            preds, labels, loss = classifier.training_step(batch)
            train_preds.append(torch.softmax(preds[:, :, 0], dim=-1))
            train_labels.append(labels)
            train_loss += (loss.item() / batch_size)
        train_preds = torch.concat(train_preds)
        if num_classes == 2:
            train_preds = train_preds[:, -1]
        train_labels = torch.concat(train_labels)
        for batch in data_loader_val:
            preds, labels, loss, _ = classifier.validation_step(batch, softmax=True)
            val_preds.append(preds[:, :, 0])
            val_labels.append(labels)
            val_loss += (loss.item() / batch_size)
        val_preds = torch.concat(val_preds)
        if num_classes == 2:
            val_preds = val_preds[:, -1]
        val_labels = torch.concat(val_labels)
        train_auc = roc_auc_score(train_labels.detach().cpu().numpy(),
                                  train_preds.detach().cpu().numpy(), multi_class='ovr')
        val_auc = roc_auc_score(val_labels.detach().cpu().numpy(),
                                val_preds.detach().cpu().numpy(), multi_class='ovr')
        print(f"Epoch {epoch + 1}: "
              f"Loss train {train_loss:.3f}, val {val_loss:.3f},  "
              f"AUC train {train_auc:.3f}, val {val_auc:.3f}")
        if epoch >= warmup:
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                checkpoint_best = copy.deepcopy(classifier.model.state_dict())
                metrics_best = {
                    'epoch': epoch + 1,
                    'train': {'loss': train_loss, 'auc': train_auc},
                    'val': {'loss': val_loss, 'auc': val_auc}
                }
                stop_count = 0
            elif val_loss >= min_val_loss + tolerance:
                stop_count += 1
            if stop_count >= patience:
                print("Early stopping.")
                break

    if checkpoint == 'best':
        res_metrics = metrics_best
        classifier.model.load_state_dict(checkpoint_best)
    else:
        res_metrics = {
            'epoch': epoch,
            'train': {'loss': train_loss, 'auc': train_auc},
            'val': {'loss': val_loss, 'auc': val_auc}
        }

    return res_metrics


def ndgcn(evidence, scores, n=None, idcg_1=False):
    if n is None:
        n = len(scores)
    sorted_evidence = evidence[np.argsort(-scores)][:n]
    if idcg_1:
        ideal_sorted_evidence = 1
    else:
        ideal_sorted_evidence = -np.sort(-evidence)[:n]
    log_indices = np.log2(np.arange(2, n + 2))
    res = (sorted_evidence / log_indices).sum() / (ideal_sorted_evidence / log_indices).sum()
    return res


def evaluate_explanation(xmodel, classifier, data_loader_test, explanation_type, evaluated_classes='label_class',
                         ndgcn_n=None):

    scores = {
        'auroc_pos': [], 'auprc_pos': [], 'ndgcn': [], 'pearsonr': [], 'spearmanr': [], 'auroc_2': [], 'auprc_2': []
    }
    all_preds, all_labels = [], []

    for batch in tqdm(data_loader_test):

        preds, label, _, _ = classifier.validation_step(batch, softmax=True)
        preds, label = preds[0, :, 0], label[0, 0]

        all_preds.append(preds)
        all_labels.append(label)

        if evaluated_classes == 'all_classes':
            eval_classes = list(batch['evidence'].keys())
        elif evaluated_classes == 'label_class':
            eval_classes = [label.item()]
        elif evaluated_classes == 'predicted_class':
            eval_classes = [preds.argmax().item()]
        elif isinstance(evaluated_classes, int):
            eval_classes = [evaluated_classes]
        elif isinstance(evaluated_classes, list):
            eval_classes = evaluated_classes
        else:
            raise ValueError(f"Unknwon evaluated classes: {evaluated_classes}")

        for eval_class in eval_classes:

            xmodel.explained_class = eval_class
            patch_scores = xmodel.get_heatmap(batch, explanation_type, False)
            evidence = batch['evidence'][eval_class][0]
            assert -1 <= evidence.min() and evidence.max() <= 1
            evidence_pos = torch.nn.functional.relu(evidence)
            evidence_neg = torch.nn.functional.relu(-evidence)
            evidence_rank = evidence + 1

            auroc_2, auprc_2 = [], []

            if 0 < evidence_pos.sum() < len(evidence_pos):
                auroc = roc_auc_score(evidence_pos.detach().cpu().numpy(), patch_scores)
                auprc = average_precision_score(evidence_pos.detach().cpu().numpy(), patch_scores)
                scores['auroc_pos'].append(auroc)
                scores['auprc_pos'].append(auprc)
                auroc_2.append(auroc)
                auprc_2.append(auprc)
                scores['ndgcn'].append(ndgcn(evidence_rank.detach().cpu().numpy(), patch_scores, ndgcn_n))
                scores['pearsonr'].append(pearsonr(evidence.detach().cpu().numpy(), patch_scores))
                scores['spearmanr'].append(spearmanr(evidence.detach().cpu().numpy(), patch_scores))

            if 0 < evidence_neg.sum() < len(evidence_neg):
                auroc_2.append(roc_auc_score(evidence_neg.detach().cpu().numpy(), -patch_scores))
                auprc_2.append(average_precision_score(evidence_neg.detach().cpu().numpy(), -patch_scores))

            if len(auroc_2) > 0:
                scores['auroc_2'].append(np.asarray(auroc_2).mean())
                scores['auprc_2'].append(np.asarray(auprc_2).mean())

    all_preds, all_labels = torch.stack(all_preds), torch.stack(all_labels)
    scores = {key: torch.tensor(val) for key, val in scores.items()}

    return all_labels, all_preds, scores
