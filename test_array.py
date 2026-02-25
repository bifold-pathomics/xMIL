import argparse
import os
import json
from test import main as test
from itertools import product
from xai.lrp_utils import set_lrp_params


def read_from_json(file_path):
    with open(file_path, "r") as file:
        loaded_data = json.load(file)
    return loaded_data


def get_args():
    parser = argparse.ArgumentParser()
    # save the model directories in a file and pass the path to this script
    parser.add_argument("--model-dirs-file", type=str, default=None)
    parser.add_argument("--model-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--save-folder-name", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default=None)

    parser.add_argument(
        "--test-checkpoint", type=str, default=None, choices=[None, "best", "last"]
    )
    parser.add_argument("--split-path", type=str, default=None)
    parser.add_argument("--metadata-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--patches-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--features-dirs", type=str, nargs="+", default=None)

    # Dataset args
    parser.add_argument(
        "--test-subsets",
        default=None,
        nargs="+",
        type=str,
        help="Split subsets that are used for testing.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        type=str,
        default=None,
        help="The target labels to predict.",
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
            "grad2",
            "ig",
            "perturbation_keep",
            "perturbation_drop",
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
        nargs="+",
        default=["logit"],
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

    parser.add_argument("--transmil-features-dim", type=str, default="features_dim")
    parser.add_argument(
        "--remove-out-layer",
        action="store_true",
        help="a temporary option--it is due to the bugfix in TransMIL for applying mlp_layers to the "
        "data. This it srelevant for models trained with an older version of the model and with "
        "nonzero n_out_layers. Turning this option on will remove the mlp_layers from TransMIL, "
        "so that LRP works correctly.",
    )

    # Environment args
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    if args.model_dirs_file is None and args.model_dirs is None:
        raise ValueError(
            f"--model-dirs and --model-dirs-file cannot be None at the same time!"
        )

    args.lrp_params = set_lrp_params(args.lrp_params)

    if len(args.explained_rel) == 1:
        args.explained_rel = args.explained_rel * len(args.explanation_types)
    elif len(args.explained_rel) != len(args.explanation_types):
        raise ValueError(
            "explained-rel must have the same number of elements as explanation-types.",
            f"They have {len(args.explained_rel)} and {len(args.explanation_types)} elements, respectively",
            "Either pass one value to explained-rel and it will be taken for all the explanation-types,",
            "or pass one explained-rel per each element explanation-types",
        )

    return args


def main():
    args = get_args()
    array_id = os.environ["SLURM_ARRAY_TASK_ID"]
    array_id = int(array_id) if array_id is not None else 0
    print("array ID: ", array_id)
    if args.model_dirs_file is not None:
        args.model_dirs = read_from_json(args.model_dirs_file)

    hopt_combinations = list(
        product(args.model_dirs, zip(args.explanation_types, args.explained_rel))
    )

    args.model_dir, (explanation_type, args.explained_rel) = hopt_combinations[array_id]
    args.explanation_types = [explanation_type]

    print(f"test script for method {explanation_type} and model_dir={args.model_dir}")

    if args.results_dir is None:
        if explanation_type == "lrp":
            folder_name = f'lrp_gamma_{args.lrp_params["gamma"]}_bias_{not args.lrp_params["no_bias"]}'
        else:
            folder_name = explanation_type
        args.save_folder_name = (
            f"explanations_{folder_name}"
            if args.save_folder_name is None
            else f"{args.save_folder_name}_{folder_name}"
        )

        args.results_dir = os.path.join(args.model_dir, args.save_folder_name)
    else:
        args.results_dir = os.path.join(args.results_dir, str(array_id))

    test(args)


if __name__ == "__main__":
    main()
