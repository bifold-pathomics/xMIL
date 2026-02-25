import os
import json
import argparse

import pandas as pd

from splits import get_label_mapping, split


def get_args():
    parser = argparse.ArgumentParser()
    # Loading and saving
    parser.add_argument("--metadata-paths", type=str, nargs="+", required=True)
    parser.add_argument("--save-path", type=str, required=True)
    # Splitting args
    parser.add_argument("--split-by", type=str, required=True)
    parser.add_argument("--data-filters", default=None)
    parser.add_argument("--targets", type=str, nargs="+")
    parser.add_argument("--groups", type=str, default=None)
    parser.add_argument(
        "--strategy",
        type=str,
        default="train_test",
        choices=["train_test", "train_val_test", "cross_validation"],
    )
    parser.add_argument(
        "--ratios",
        default=None,
        help="Dict for the split ratios for the chosen strategy",
    )
    parser.add_argument("--label-threshold", type=float, default=0.5)
    parser.add_argument("--balance-subsets", default=None, choices=["train", "all"])
    parser.add_argument("--balance-group", default=None)
    parser.add_argument("--survival", action="store_true")
    parser.add_argument("--continuous-target", action="store_true")
    parser.add_argument(
        "--censorship-mapping",
        default=None,
        help="Dict defining the mapping of survival status to binary.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.survival:  # make sure that for survival the continuous_target is turned on
        args.continuous_target = True

    # Parse dict-like args
    if args.data_filters is not None:
        args.data_filters = json.loads(args.data_filters)

    if args.ratios is not None:
        args.ratios = json.loads(args.ratios)

    if args.censorship_mapping is not None:
        args.censorship_mapping = json.loads(args.censorship_mapping)
        args.censorship_mapping = {
            int(k): v for k, v in args.censorship_mapping.items()
        }  # ensure the keys are integer
    else:
        args.censorship_mapping = {"0:LIVING": 1, "1:DECEASED": 0}

    if not args.continuous_target and len(args.targets) == 1:
        args.targets = args.targets[0]
    return args


def main():
    # Read args
    args = get_args()
    print(json.dumps(vars(args), indent=4))
    if os.path.exists(args.save_path):
        raise ValueError(f"Target file already exists: {args.save_path}")
    print(f"Results will be written to: {args.save_path}")
    # Get label mapping
    if (
        not args.continuous_target
        and args.targets is not None
        and isinstance(args.targets, str)
    ):
        label_mapping = get_label_mapping(args.targets, args.label_threshold)
    else:
        label_mapping = None
    # Read and merge metadata
    metadata = pd.DataFrame()
    for idx, metadata_path in enumerate(args.metadata_paths):
        metadata = pd.concat(
            [metadata, pd.read_csv(metadata_path)], axis=0, ignore_index=True
        )
    # Filter metadata
    if args.data_filters is not None:
        for key, vals in args.data_filters.items():
            metadata = metadata[metadata[key].isin(vals)]
    # Compute and save split
    split_df = split(
        metadata=metadata,
        split_by=args.split_by,
        target=args.targets,
        label_mapping=label_mapping,
        groups=args.groups,
        strategy=args.strategy,
        ratios=args.ratios,
        balance_subsets=args.balance_subsets,
        balance_group=args.balance_group,
        seed=args.seed,
        survival=args.survival,
        continuous_target=args.continuous_target,
        censorship_mapping=args.censorship_mapping,
    )
    split_df.to_csv(args.save_path, index=False)


if __name__ == "__main__":
    main()
