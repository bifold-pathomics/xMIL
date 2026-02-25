import argparse
import os
import json
from visualize import main as visualize


def read_from_json(file_path):
    with open(file_path, "r") as file:
        loaded_data = json.load(file)
    return loaded_data


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dirs-file", type=str, required=True)
    """
    save the model directories in a file and pass the path to this script
    """

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
        "--scores-dir-root",
        type=str,
        required=True,
        help="the root directory containing the direcotories including the patch scores.",
    )
    parser.add_argument("--heatmap-type", type=str, default="lrp")
    parser.add_argument("--lrp-explained-class", type=int, default=1)
    parser.add_argument("--cut-top-k-percent", type=int, default=None)
    parser.add_argument("--no-side-by-side", action="store_true")
    parser.add_argument("--no-show-annotations", action="store_true")
    parser.add_argument("--no-top-patches", action="store_true")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--no-label", action="store_true")

    # Environment args
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    array_id = os.environ["SLURM_ARRAY_TASK_ID"]
    array_id = int(array_id) if array_id is not None else 0
    print("array ID: ", array_id)
    args.pdf_suffix = f"_{array_id}"

    args.model_dirs = read_from_json(args.model_dirs_file)
    args.model_dir = args.model_dirs[array_id]
    print("model_dir: ", args.model_dir)

    args.explain_scores_path = os.path.join(
        args.scores_dir_root, str(array_id), "test_predictions.csv"
    )
    print("explain_scores_path: ", args.explain_scores_path)

    args.results_dir = os.path.join(args.results_dir, str(array_id))

    visualize(args)


if __name__ == "__main__":
    main()
