import os
import json
import argparse
from itertools import product
import hashlib

from train import main as train


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
        default=[["train"]],
        type=json.loads,
        help="Split subsets that are used for training.",
    )
    parser.add_argument(
        "--val-subsets",
        default=[["test"]],
        type=json.loads,
        help="Split subsets that are used for validation.",
    )
    parser.add_argument(
        "--test-subsets",
        default=[None],
        type=json.loads,
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
        nargs="+",
        default=[None],
        help="Number of patches to sample per slide. If None or -1, all patches are used."
        "Zipped with train-bag-size, train-batch-size.",
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
    parser.add_argument(
        "--preload-data",
        action="store_true",
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
        "--head-dim", type=int, default=2, help="The number of classes to predict."
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
    parser.add_argument(
        "--features-dim",
        type=int,
        nargs="+",
        default=[256],
        help="Output dimension of the initial linear layer applied to the feature vectors in a model.",
    )

    parser.add_argument("--metric-name", type=str, nargs="+", default=None)
    # todo: the decision_metric should be set up for classification as well. for survival it should work with 'c_index'
    parser.add_argument(
        "--ref-value",
        type=float,
        default=None,
        help="reference value for the regression model.",
    )

    # -- Attention MIL
    parser.add_argument(
        "--inner-attention-dim",
        type=int,
        nargs="+",
        default=[128],
        help="Inner hidden dimension of the 2-layer attention mechanism in an AttentionMIL model.",
    )
    parser.add_argument(
        "--dropout-strategy",
        type=str,
        nargs="+",
        default=["features"],
        choices=["features", "last", "all"],
        help="Which layers to apply dropout to.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        nargs="+",
        default=[0],
        help="Fraction of neurons to drop per targeted layer. None to apply no dropout.",
    )
    parser.add_argument("--num-layers", type=int, nargs="+", default=[1])

    # -- TransMIL
    parser.add_argument(
        "--dropout-att",
        type=float,
        nargs="+",
        default=[0.75],
        help="Zipped with dropout-att, dropout-class, dropout-feat.",
    )
    parser.add_argument(
        "--dropout-class",
        type=float,
        nargs="+",
        default=[0.75],
        help="Zipped with dropout-att, dropout-class, dropout-feat.",
    )
    parser.add_argument(
        "--dropout-feat",
        type=float,
        nargs="+",
        default=[0],
        help="Zipped with dropout-att, dropout-class, dropout-feat.",
    )
    parser.add_argument("--attention", type=str, default="nystrom")
    parser.add_argument("--n-layers", type=int, nargs="+", default=[2])
    parser.add_argument("--no-attn-residual", action="store_true")
    parser.add_argument("--pool-method", type=str, default="cls_token")
    parser.add_argument("--no-ppeg", action="store_true")

    # -- MambaMIL
    parser.add_argument("--pos-embed", type=str, default="none")
    parser.add_argument("--scan", type=str, default="simple")

    # Training args
    parser.add_argument(
        "--train-batch-size",
        type=int,
        nargs="+",
        default=[8],
        help="Zipped with train-bag-size, train-batch-size.",
    )
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, nargs="+", default=[5e-3])
    parser.add_argument("--weight-decay", type=float, nargs="+", default=[0.001])
    parser.add_argument("--schedule-lr", action="store_true")
    parser.add_argument("--loss-type", type=str, default="cross-entropy")
    parser.add_argument("--num-epochs", type=int, nargs="+", default=[100])
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--stop-criterion", type=str, default="loss")
    parser.add_argument("--optimizer", type=str, nargs="+", default=["SGD"])
    parser.add_argument("--grad-clip", type=float, nargs="+", default=[None])
    parser.add_argument("--warmup", type=int, default=0)

    # Testing args
    parser.add_argument(
        "--test-checkpoint", type=str, default="best", choices=["best", "last"]
    )

    # Seed
    parser.add_argument(
        "--seed",
        type=int,
        nargs="+",
        default=[0],
        help="WARNING This is currently a dummy, seeding is not implemented.",
    )

    # Environment args
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--save-folder",
        type=str,
        default="task_id",
        choices=["task_id", "hashlib_sha256"],
    )
    parser.add_argument("--num-workers", type=int, default=0)

    # Parse all args
    args = parser.parse_args()

    return args


def get_hopt_combination(args):
    """
    Selects the combination of args for the current run based on the SLURM_ARRAY_TASK_ID environment variable.
    """
    array_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    hopt_combinations = list(
        product(
            list(zip(args.train_subsets, args.val_subsets, args.test_subsets)),
            list(zip(args.train_bag_size, args.train_batch_size)),
            list(zip(args.features_dim, args.inner_attention_dim)),
            args.dropout_strategy,
            args.dropout,
            args.num_layers,
            list(zip(args.dropout_att, args.dropout_class, args.dropout_feat)),
            args.n_layers,
            args.learning_rate,
            args.weight_decay,
            args.num_epochs,
            args.grad_clip,
            args.seed,
            args.optimizer,
        )
    )
    (
        (args.train_subsets, args.val_subsets, args.test_subsets),
        (args.train_bag_size, args.train_batch_size),
        (args.features_dim, args.inner_attention_dim),
        args.dropout_strategy,
        args.dropout,
        args.num_layers,
        (args.dropout_att, args.dropout_class, args.dropout_feat),
        args.n_layers,
        args.learning_rate,
        args.weight_decay,
        args.num_epochs,
        args.grad_clip,
        args.seed,
        args.optimizer,
    ) = hopt_combinations[array_id]

    if args.grad_clip is not None and args.grad_clip < 0:
        args.grad_clip = None
    if args.ref_value is not None and args.ref_value < 0:
        args.ref_value = None

    if args.save_folder == "hashlib_sha256":
        config_string = "_".join(f"{str(k)}*{str(v)}" for k, v in vars(args).items())
        hash_object = hashlib.sha256(config_string.encode())
        unique_id = hash_object.hexdigest()[:10]
    else:  # elif args.save_folder == 'task_id':
        unique_id = str(array_id)

    args.results_dir = os.path.join(args.results_dir, unique_id)
    return args


def main():
    args = get_args()
    args = get_hopt_combination(args)
    train(args)


if __name__ == "__main__":
    main()
