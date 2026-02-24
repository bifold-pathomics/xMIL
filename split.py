import os
import json
import argparse

import pandas as pd

from splits import get_label_mapping, split


def get_args():
    parser = argparse.ArgumentParser()
    # Loading and saving
    parser.add_argument("--metadata-paths", type=str, nargs="+", required=True)
    parser.add_argument("--target-path", type=str, required=True)
    # Splitting args
    parser.add_argument("--split-by", type=str, required=True)
    parser.add_argument("--data-filters", default=None)
    parser.add_argument("--target", type=str, default=None)
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    # Parse dict-like args
    if args.data_filters is not None:
        args.data_filters = json.loads(args.data_filters)
    if args.ratios is not None:
        args.ratios = json.loads(args.ratios)
    return args


def main():
    # Read args
    args = get_args()
    print(json.dumps(vars(args), indent=4))
    if os.path.exists(args.target_path):
        raise ValueError(f"Target file already exists: {args.target_path}")
    print(f"Results will be written to: {args.target_path}")
    # Get label mapping
    if args.target is not None:
        label_mapping = get_label_mapping(args.target, args.label_threshold)
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
        target=args.target,
        label_mapping=label_mapping,
        groups=args.groups,
        strategy=args.strategy,
        ratios=args.ratios,
        seed=args.seed,
    )
    split_df.to_csv(args.target_path, index=False)


if __name__ == "__main__":
    main()
