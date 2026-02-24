import os
import json
import argparse

import torch

from datasets import DatasetFactory
from models import ModelFactory, xModelFactory
from training import Callback, test_classification_model


def get_args():
    parser = argparse.ArgumentParser()

    # Loading and saving
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument(
        "--test-checkpoint", type=str, default=None, choices=[None, "best", "last"]
    )
    parser.add_argument("--split-path", type=str, default=None)
    parser.add_argument("--metadata-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--patches-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--features-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--results-dir", type=str, required=True)

    # Dataset args
    parser.add_argument(
        "--test-subsets",
        default=None,
        nargs="+",
        type=str,
        help="Split subsets that are used for testing.",
    )
    parser.add_argument(
        "--drop-duplicates", type=str, default="sample", choices=["sample", "case"]
    )
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument(
        "--patch-filters",
        type=json.loads,
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
    parser.add_argument(
        "--preload-data",
        action="store_true",
        help="Whether to preload all features into RAM before starting training.",
    )

    # Explanations
    parser.add_argument(
        "--explanation-types",
        default=None,
        nargs="+",
        type=str,
        choices=[
            None,
            "attention",
            "patch_scores",
            "lrp",
            "gi",
            "ig",
            "grad2",
            "perturbation_keep",
            "perturbation_drop",
        ],
        help="If given, patch explanation scores are computed and saved in the predictions dataframe.",
    )
    parser.add_argument(
        "--explained-class", type=int, default=None, help="The class to be explained."
    )
    parser.add_argument(
        "--explained-rel",
        type=str,
        default="logit",
        help="The type of output to be explained.",
    )
    parser.add_argument(
        "--lrp-params",
        type=json.loads,
        default=None,
        help="LRP params for LRP explanations.",
    )
    parser.add_argument(
        "--contrastive-class",
        type=int,
        default=0,
        help="The class to be explained against (if explained-rel is contrastive).",
    )
    parser.add_argument(
        "--attention-layer",
        type=int,
        default=None,
        help="For which attention layer to extract attention scores. If None, attention rollout "
        "over all layers.",
    )
    parser.add_argument("--detach-pe", action="store_true")

    # Environment args
    parser.add_argument("--device", type=str, default="cpu")

    # Parse all args
    args = parser.parse_args()

    return args


def main():
    # Process and save input args
    args = get_args()

    # Load args from model training
    with open(os.path.join(args.model_dir, "args.json")) as f:
        model_args = json.load(f)

    # replace parameters if needed
    for param_name, param_value in model_args.items():
        if (
            (param_name == "split_path" and args.split_path is None)
            or (param_name == "metadata_dirs" and args.metadata_dirs is None)
            or (param_name == "patches_dirs" and args.patches_dirs is None)
            or (param_name == "features_dirs" and args.features_dirs is None)
        ):
            setattr(args, param_name, param_value)

    print(json.dumps(vars(args), indent=4))

    save_dir = args.results_dir
    os.makedirs(save_dir, exist_ok=False)
    print(f"Results will be written to: {save_dir}")
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Set up environment
    device = torch.device(args.device)

    # Load dataset structures
    _, _, _, _, test_dataset, test_loader = DatasetFactory.build(vars(args), model_args)

    # Set up callback, model, and load model weights
    callback = Callback(
        schedule_lr=None,
        checkpoint_epoch=1,
        path_checkpoints=args.model_dir,
        early_stop=False,
        device=device,
        results_dir=save_dir,
    )

    model, classifier = ModelFactory.build(model_args, device)

    print(f"Loading model into RAM from: {args.model_dir}")
    checkpoint = (
        args.test_checkpoint
        if args.test_checkpoint is not None
        else model_args["test_checkpoint"]
    )
    model = callback.load_checkpoint(model, checkpoint=checkpoint)

    # Set up explanation model if desired
    if args.explanation_types is not None:
        xmodel = xModelFactory.build(model, vars(args))
    else:
        xmodel = None

    # Run test

    print(f"Test set evaluation with checkpoint: {checkpoint}")
    test_classification_model(
        model=model,
        classifier=classifier,
        dataloader_test=test_loader,
        callback=callback,
        label_cols=model_args.get("targets", ["label"]),
        xmodel=xmodel,
        explanation_types=args.explanation_types,
        tb_writer=None,
        verbose=False,
    )


if __name__ == "__main__":
    main()
