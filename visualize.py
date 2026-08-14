import os
import json
import argparse

import PIL
import torch
import pandas as pd

from datasets import DatasetFactory
from models import ModelFactory, xModelFactory
from training import Callback
from visualization import create_visualization_pdf

PIL.Image.MAX_IMAGE_PIXELS = 196455024 * 10


def get_args():
    parser = argparse.ArgumentParser()

    # Loading and saving
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument(
        "--model-checkpoint", type=str, default=None, choices=[None, "best", "last"]
    )
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--slides-dirs", type=str, nargs="+", required=True)
    parser.add_argument("--annotations-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--local-split", action="store_true")
    parser.add_argument(
        "--preload-data",
        action="store_true",
        help="Whether to preload all features into RAM before starting training.",
    )

    # Optional dataset args to overwrite model args
    parser.add_argument(
        "--targets",
        nargs="+",
        type=str,
        default=["label"],
        help="The target labels to predict.",
    )
    parser.add_argument("--split-path", type=str, default=None)
    parser.add_argument("--metadata-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--patches-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--features-dirs", type=str, nargs="+", default=None)
    parser.add_argument(
        "--test-subsets",
        default=None,
        nargs="+",
        type=str,
        help="Split subsets that are used for testing.",
    )
    parser.add_argument(
        "--drop-duplicates", type=str, default=None, choices=[None, "sample", "case"]
    )
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument(
        "--patch-filters",
        default=None,
        help="Filters to only use a selected subset of patches per slide."
        "Pass {'has_annot': [1, 2]} to only use patches with some annotation of class 1 or 2."
        "Pass {'exclude_annot': [0, 8]} to only use patches with no annotation of class 0 and 8.",
    )
    parser.add_argument(
        "--max-bag-size",
        type=int,
        default=None,
        help="Maximum number of patches per slide. Slides with more patches are dropped.",
    )

    # Explanation args
    parser.add_argument(
        "--explain-scores-path",
        type=str,
        default=None,
        help="the path to the csv file containing the patch scores.",
    )
    parser.add_argument("--heatmap-type", type=str, default="lrp")
    parser.add_argument("--lrp-explained-class", type=int, default=1)
    parser.add_argument("--cut-top-k-percent", type=int, default=None)
    parser.add_argument("--no-side-by-side", action="store_true")
    parser.add_argument("--no-show-annotations", action="store_true")
    parser.add_argument("--no-top-patches", action="store_true")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--no-label", action="store_true")

    parser.add_argument("--pdf-suffix", type=str, default="")
    # Environment args
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    return args


def main(args=None):
    # Read user and model args
    args = get_args() if args is None else args
    with open(os.path.join(args.model_dir, "args.json")) as f:
        model_args = json.load(f)
    model_type = model_args["aggregation_model"]
    if args.local_split:
        model_args["split_path"] = os.path.join(args.model_dir, "split.csv")

    # Set up environment
    device = torch.device(args.device)
    print(json.dumps(vars(args), indent=4))

    # backwards compatibility
    try:
        pred_targets = model_args["targets"]
    except KeyError:
        print(
            "The model arguments does not contain target variable. We set to the default targets ['label']"
        )
        pred_targets = ["label"]

    # Set up dataset structures
    dataset_args = model_args
    for key, val in vars(args).items():
        if val is not None:
            dataset_args[key] = val

    dataset_args["train_subsets"] = None
    dataset_args["val_subsets"] = None

    print(dataset_args["split_path"])
    _, _, _, _, test_dataset, test_loader = DatasetFactory.build(
        dataset_args, model_args
    )

    # Set up model and classifier structures
    callback = Callback(
        schedule_lr=None,
        checkpoint_epoch=1,
        path_checkpoints=args.model_dir,
        early_stop=False,
        device=device,
    )

    model, classifier = ModelFactory.build(model_args, device)

    print(f"Loading model into RAM from: {args.model_dir}")
    checkpoint = (
        args.model_checkpoint
        if args.model_checkpoint is not None
        else model_args["test_checkpoint"]
    )
    model = callback.load_checkpoint(model, checkpoint=checkpoint)

    # Set up explanation model
    xmodel = xModelFactory.build(model, vars(args))

    if args.explain_scores_path is not None:
        df_predictions = pd.read_csv(args.explain_scores_path)
    else:
        df_predictions = None

    # Run visualization code
    create_visualization_pdf(
        data_loader=test_loader,
        classifier=classifier,
        xmodel=xmodel,
        slides_dirs=args.slides_dirs,
        patches_dirs=model_args["patches_dirs"],
        results_dir=args.results_dir,
        scores_df=df_predictions,
        annotations_dirs=args.annotations_dirs,
        side_by_side=(not args.no_side_by_side),
        print_label=(not args.no_label),
        show_annotations=(not args.no_show_annotations),
        heatmap_type=args.heatmap_type,
        top_patches=(not args.no_top_patches),
        save_overlays=(not args.no_overlays),
        target_names=pred_targets,
        cut_top_k_percent=args.cut_top_k_percent,
        pdf_suffix=args.pdf_suffix,
    )


if __name__ == "__main__":
    main()
