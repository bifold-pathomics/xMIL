import os
import json
import argparse
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from xmil.splits import balance_labels
from xmil.datasets import DatasetFactory
from xmil.models import ModelFactory
from xmil.training import TrainTestExecutor, Callback


def get_args():
    parser = argparse.ArgumentParser()

    # Modalities
    parser.add_argument(
        "--mode", type=str, default="image-only", choices=["image-only", "omics-only"]
    )

    # Loading and saving
    parser.add_argument("--split-path", type=str, required=True)
    parser.add_argument("--metadata-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--patches-dirs", type=str, nargs="+")
    parser.add_argument("--features-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--results-dir", type=str, required=True)

    # Dataset args
    parser.add_argument(
        "--train-subsets",
        default=["train"],
        nargs="+",
        type=str,
        help="Split subsets that are used for training.",
    )
    parser.add_argument(
        "--val-subsets",
        default=["test"],
        nargs="+",
        type=str,
        help="Split subsets that are used for validation.",
    )
    parser.add_argument(
        "--test-subsets",
        default=None,
        nargs="+",
        type=str,
        help="Split subsets that are used for testing.",
    )
    parser.add_argument(
        "--balance-key",
        type=str,
        default=None,
        help="balances train dataset with respect to the given key",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=None,
        choices=[None, "bts"],
        help="Sampling strategy for the training data. If None, random shuffling will be used.",
    )
    parser.add_argument(
        "--drop-duplicates", type=str, default="sample", choices=["sample", "case"]
    )
    parser.add_argument(
        "--patch-filters",
        type=json.loads,
        default=None,
        help="Filters to only use a selected subset of patches per slide."
        "Pass {'has_annot': [1, 2]} to only use patches with some annotation of class 1 or 2."
        "Pass {'exclude_annot': [0, 8]} to only use patches with no annotation of class 0 and 8.",
    )
    parser.add_argument(
        "--train-bag-size",
        type=int,
        default=None,
        help="Number of patches to sample per slide. If None or -1, all patches are used.",
    )
    parser.add_argument(
        "--sort-sampled-patches",
        type=lambda x: (str(x).lower() == "true"),
        nargs="?",
        const=True,
        default=False,
        help="Whether to sort sampled patches.",
    )
    parser.add_argument(
        "--test-bag-sampling",
        action="store_true",
        help="Whether to sample bags at test time with the same bag size as during training. If not,"
        "all patches of a slide are considered.",
    )
    parser.add_argument(
        "--test-repetitions",
        type=int,
        default=1,
        help="How many times to predict the same slide. Relevant if test-bag-sampling is True.",
    )
    parser.add_argument(
        "--max-bag-size",
        type=int,
        default=None,
        help="Maximum number of patches per slide. Slides with more patches are dropped.",
    )
    parser.add_argument("--min-bag-size", type=int, default=0)
    parser.add_argument(
        "--preload-data",
        type=lambda x: (str(x).lower() == "true"),
        nargs="?",
        const=True,
        default=False,
        help="Whether to preload all features into RAM before starting training.",
    )

    # Model args
    parser.add_argument(
        "--aggregation-model",
        type=str,
        default="attention_mil",
        choices=["attention_mil", "transmil", "additive_mil", "mamba_mil"],
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=2048,
        help="The dimension of the feature vectors.",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=2,
        help="The number of outputs (logits) of the last linear layer of the models.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        type=str,
        default=["label"],
        help="The target labels to predict.",
    )
    parser.add_argument("--no-bias", action="store_true")
    parser.add_argument("--n-out-layers", type=int, default=0)
    parser.add_argument(
        "--num-encoders",
        type=int,
        default=1,
        help="How many separate instance encoders to use for a bag. For one encoder for all bag "
        "instances, pass 1. For a separate encoder per instance, pass num_instances_per_bag.",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default="classification",
        choices=["survival", "classification", "regression"],
    )

    parser.add_argument("--metric-name", type=str, nargs="+", default=None)
    # todo: the decision_metric should be set up for classification as well. for survival it should work with 'c_index'
    parser.add_argument(
        "--ref-value",
        type=float,
        default=None,
        help="reference value for the regression model.",
    )

    parser.add_argument(
        "--features-dim",
        type=int,
        default=256,
        help="Output dimension of the initial linear layer applied to the feature vectors in a model.",
    )

    # -- Attention MIL
    parser.add_argument(
        "--inner-attention-dim",
        type=int,
        default=128,
        help="Inner hidden dimension of the 2-layer attention mechanism in an AttentionMIL model.",
    )
    parser.add_argument(
        "--dropout-strategy",
        type=str,
        default="features",
        choices=["features", "last", "all"],
        help="Which layers to apply dropout to.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Fraction of neurons to drop per targeted layer. None to apply no dropout.",
    )
    parser.add_argument("--num-layers", type=int, default=1)

    # -- TransMIL
    parser.add_argument("--dropout-att", type=float, default=0.75)
    parser.add_argument("--dropout-class", type=float, default=0.75)
    parser.add_argument("--dropout-feat", type=float, default=0)
    parser.add_argument("--attention", type=str, default="nystrom")
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--no-attn-residual", action="store_true")
    parser.add_argument("--pool-method", type=str, default="cls_token")
    parser.add_argument("--no-ppeg", action="store_true")

    # -- MambaMIL
    parser.add_argument("--pos-embed", type=str, default="none")
    parser.add_argument("--scan", type=str, default="simple")

    # Training args
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--schedule-lr", action="store_true")
    parser.add_argument("--loss-type", type=str, default="cross-entropy")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--stop-criterion", type=str, default="loss")
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=0)

    # Testing args
    parser.add_argument(
        "--test-checkpoint", type=str, default="best", choices=["best", "last"]
    )

    # Environment args
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)

    # Parse all args
    args = parser.parse_args()

    if args.grad_clip is not None and args.grad_clip < 0:
        args.grad_clip = None
    if args.ref_value is not None and args.ref_value < 0:
        args.ref_value = None

    return args


def main(args=None):
    # Process args and create file structures
    args = get_args() if args is None else args

    print(json.dumps(vars(args), indent=4))
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    save_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(save_dir)
    print(f"Results will be written to: {save_dir}")
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Set up environment
    device = torch.device(args.device)
    tb_writer = SummaryWriter(save_dir)

    # Apply split balancing if requested
    if args.balance_key:
        df_split = pd.read_csv(args.split_path)
        df_split["subset"] = df_split["subset"].astype(str)
        df_metadata_case = pd.DataFrame()
        for idx, metadata_dir in enumerate(args.metadata_dirs):
            df_ = pd.read_csv(os.path.join(metadata_dir, "case_metadata.csv"))
            df_.insert(0, "source_id", idx)
            df_metadata_case = pd.concat(
                [df_metadata_case, df_], axis=0, ignore_index=True
            )
        df_metasplit = pd.merge(
            df_split, df_metadata_case[["case_id", args.balance_key]], on="case_id"
        )
        # todo: balance labels is not tested for survival
        df_split_bal = balance_labels(
            df_metasplit,
            label_cols=args.targets,
            subsets=args.train_subsets,
            group_key=args.balance_key,
            strategy="drop",
            seed=None,
        )[df_split.keys()]
        args.split_path = os.path.join(save_dir, "split.csv")
        df_split_bal.to_csv(args.split_path)

    # Set up dataset structures
    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
        DatasetFactory.build(vars(args), vars(args))
    )
    if args.head_type == "survival":
        np.save(
            os.path.join(save_dir, "survival_bins.npy"), train_dataset.survival_bins
        )

    if args.head_type == "regression":
        if args.ref_value is None:
            args.ref_value = (
                train_dataset.split_metadata[train_dataset.label_cols].median().item()
            )
        with open(os.path.join(save_dir, "ref_value.json"), "w") as f:
            json.dump(args.ref_value, f)

    # Set up model and classifier
    model, classifier = ModelFactory.build(vars(args), device)

    # Set up callback
    callback = Callback(
        schedule_lr=args.schedule_lr,
        checkpoint_epoch=args.val_interval,
        path_checkpoints=save_dir,
        stop_criterion=args.stop_criterion,
        early_stop=args.early_stopping,
    )

    learner = TrainTestExecutor(
        model=model, callback=callback, model_args=vars(args), explanation_args=None
    )

    learner.train(
        train_loader=train_loader,
        val_loader=val_loader,
        classifier=classifier,
        tb_writer=tb_writer,
    )

    print(f"Test set evaluation with checkpoint: {args.test_checkpoint}")
    learner.test(
        test_loader=test_loader,
        classifier=classifier,
        xmodel=None,
        tb_writer=tb_writer,
        checkpoint=args.test_checkpoint,
    )

    # Clean up
    tb_writer.close()
    print("Finished model training")


if __name__ == "__main__":
    main()
