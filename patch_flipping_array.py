import json
import argparse
import os
from itertools import product
from evaluation_patch_flipping import main as evaluation_patch_flipping
from xai.lrp_utils import set_lrp_params


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", type=str, default="image-only", choices=["image-only", "omics-only"]
    )

    # Loading and saving
    parser.add_argument("--model-dirs-file", type=str, required=True)
    parser.add_argument("--model-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--predictions-dirs", type=str, nargs="+", default=None)
    parser.add_argument("--sel-checkpoint", type=str, default="best")
    parser.add_argument(
        "--dataset",
        type=str,
        default="test",
        help="the dataset for which the patch dropping is performed. can be train, val, or test.",
    )
    parser.add_argument("--explanation-types", type=str, nargs="+", required=True)
    parser.add_argument(
        "--prediction-file-name", type=str, default="test_predictions.csv"
    )

    parser.add_argument("--max-bag-size", type=int, default=-1)
    parser.add_argument("--min-bag-size", type=int, default=0)

    # Analyses
    parser.add_argument("--strategy", type=str, default="remaining-10-perc")
    parser.add_argument("--approach", type=str, default="drop", choices=["drop", "add"])
    parser.add_argument(
        "--attribution-strategy", type=str, nargs="+", default=["original"]
    )
    parser.add_argument("--order", type=str, nargs="+", default=["descending"])

    # Explanations
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
        default=None,
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
    parser.add_argument("---preload-data", action="store_true")
    parser.add_argument(
        "--remove-out-layer",
        action="store_true",
        help="this option is a temporary option--it is due to the bugfix in transmil.",
    )

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()

    if args.model_dirs_file.lower() == "none":
        args.model_dirs_file = None
    if args.model_dirs_file is None and args.model_dirs is None:
        raise ValueError(
            "args.model_dirs_file=None and args.model_dirs=None. No model is given."
        )

    args.lrp_params = set_lrp_params(args.lrp_params)
    return args


def main():
    args = get_args()
    array_id = os.environ["SLURM_ARRAY_TASK_ID"]
    array_id = int(array_id) if array_id is not None else 0
    print("array ID: ", array_id)
    if args.model_dirs_file is not None:
        with open(args.model_dirs_file, "r") as file:
            args.model_dirs = json.load(file)

    if args.predictions_dirs is None:
        args.predictions_dirs = args.model_dirs
    elif len(args.predictions_dirs) != len(args.model_dirs):
        raise ValueError(
            "the length of predictions_dirs should be the same as model_dirs,",
            "or pass None if the predictions are saved with the model directories.",
        )

    hopt_combinations = list(
        product(zip(args.model_dirs, args.predictions_dirs), args.explanation_types)
    )
    (args.model_path, args.predictions_dir), explanation_type = hopt_combinations[
        array_id
    ]
    args.explanation_types = [explanation_type]

    if explanation_type == "lrp":
        folder_name = f'lrp_gamma_{args.lrp_params["gamma"]}_bias_{not args.lrp_params["no_bias"]}'
    else:
        folder_name = explanation_type

    explain_scores_dir = os.path.join(
        args.predictions_dir, f"explanations_{folder_name}"
    )
    args.predictions_path = os.path.join(explain_scores_dir, args.prediction_file_name)
    args.precomputed_heatmap_types = [explanation_type]
    args.results_dir = explain_scores_dir

    evaluation_patch_flipping(args)


if __name__ == "__main__":
    main()
