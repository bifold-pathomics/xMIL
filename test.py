import os
import json
import argparse

import numpy as np
import torch

from datasets import DatasetFactory
from models import ModelFactory, xModelFactory
from training import TrainTestExecutor, Callback
from xai.lrp_utils import set_lrp_params


def make_backward_compatible(model_args, args):
    """
    Modifies the model arguments to match the current arguments.
    :param model_args: [dict] model_args
    :param args: input args
    :return:
    """
    if "head_dim" not in model_args:
        model_args["head_dim"] = model_args.get("num_classes", model_args["n_out"])

    model_args["features_dim"] = model_args[args.transmil_features_dim]

    if "mode" not in model_args:
        model_args["mode"] = args.mode

    model_args["head_type"] = model_args.get("head_type", "classification")
    return model_args


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", type=str, default="image-only", choices=["image-only", "omics-only"]
    )

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
    parser.add_argument(
        "--targets",
        nargs="+",
        type=str,
        default=None,
        help="The target labels to predict.",
    )
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
    parser.add_argument("--min-bag-size", type=int, default=0)
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
            "grad2",
            "perturbation_keep",
            "perturbation_drop",
            "ig",
            "random",
        ],
        help="If given, patch explanation scores are computed and saved in the predictions dataframe.",
    )
    parser.add_argument("--save-vectors", action="store_true")
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
    parser.add_argument("--not-compute-auc", action="store_true")
    parser.add_argument("--overwrite-prev-results", action="store_true")

    parser.add_argument(
        "--remove-out-layer",
        action="store_true",
        help="a temporary option--it is due to the bugfix in TransMIL for applying mlp_layers to the "
        "data. This it srelevant for models trained with an older version of the model and with "
        "nonzero n_out_layers. Turning this option on will remove the mlp_layers from TransMIL, "
        "so that LRP works correctly.",
    )
    parser.add_argument(
        "--transmil-features-dim",
        type=str,
        default="features_dim",
        choices=["num_features", "features_dim"],
        help="num_features is relevant for the models trained on previous version",
    )

    # Environment args
    parser.add_argument("--device", type=str, default="cpu")

    # Parse all args
    args = parser.parse_args()

    args.lrp_params = set_lrp_params(args.lrp_params)

    return args


def main(args=None):
    # Process and save input args
    if args is None:
        args = get_args()

    # Load args from model training
    with open(os.path.join(args.model_dir, "args.json")) as f:
        model_args = json.load(f)

    model_args = make_backward_compatible(model_args, args)

    head_type = model_args.get("head_type", "classification")
    model_args["head_type"] = head_type

    # Replace parameters if needed
    for param_name, param_value in model_args.items():
        if (
            (param_name == "split_path" and args.split_path is None)
            or (param_name == "metadata_dirs" and args.metadata_dirs is None)
            or (param_name == "patches_dirs" and args.patches_dirs is None)
            or (param_name == "features_dirs" and args.features_dirs is None)
        ):
            setattr(args, param_name, param_value)

    if args.test_subsets is None:
        args.test_subsets = model_args["test_subsets"]

    print(json.dumps(vars(args), indent=4))

    # Set the save directory
    save_dir = args.results_dir
    os.makedirs(save_dir, exist_ok=args.overwrite_prev_results)
    print(f"Results will be written to: {save_dir}")
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # todo: if heatmap_type=='random' then skip loading the datasets and model predictions.
    #  and directly create the test_predictions.csv
    # Complete dataset_args
    args_dataset = {
        **model_args,
        **{key: val for key, val in vars(args).items() if val is not None},
    }
    if head_type == "survival":
        if not os.path.exists(os.path.join(args.model_dir, "survival_bins.npy")):
            raise FileNotFoundError(
                "survival_bins.npy not found in the model directory."
            )
        args_dataset["survival_bins"] = np.load(
            os.path.join(args.model_dir, "survival_bins.npy")
        )
    else:
        args_dataset["survival_bins"] = None

    if head_type == "regression":
        if not os.path.exists(os.path.join(args.model_dir, "ref_value.json")):
            model_args["ref_value"] = None
        else:
            with open(os.path.join(args.model_dir, "ref_value.json"), "r") as f:
                model_args["ref_value"] = json.load(f)

    if args.targets is None:
        args_dataset["targets"] = model_args["targets"]
    args_dataset["targets"] = (
        args.targets if args.targets is not None else model_args["targets"]
    )
    args_dataset["head_type"] = head_type
    args_dataset["train_subsets"], args_dataset["val_subsets"] = None, None

    # Set up environment
    device = torch.device(args.device)

    # Load dataset structures
    _, _, _, _, test_dataset, test_loader = DatasetFactory.build(
        args_dataset, model_args
    )

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
    if args.remove_out_layer:
        model.set_out_layers(torch.nn.Sequential(*[]))

    # Set up explanation model if desired
    explanation_args = vars(args)
    explanation_args["head_type"] = head_type

    print("explanation_args: ")
    print(json.dumps(explanation_args, indent=4))

    if args.explanation_types is not None:
        xmodel = xModelFactory.build(model, explanation_args)
    else:
        xmodel = None

    learner = TrainTestExecutor(
        model=model,
        callback=callback,
        model_args=model_args,
        explanation_args=explanation_args,
    )

    print(f"Test set evaluation with checkpoint: {checkpoint}")
    learner.test(
        test_loader, classifier, xmodel=xmodel, tb_writer=None, checkpoint=None
    )


if __name__ == "__main__":
    main()
