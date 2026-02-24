import os
import json
import argparse
from datetime import datetime
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from toy_experiments.datasets import (
    get_number_mil_dataset,
    bag_collate_fn,
    get_MNIST_features,
)
from toy_experiments.scripts import train_model, evaluate_explanation
from toy_experiments.models import get_model_and_classifier, get_xmodel


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--results-dir", type=str, required=True)

    parser.add_argument("--num-repetitions", type=int, default=1)

    parser.add_argument("--dataset-type", type=str, required=True)
    parser.add_argument("--num-numbers", type=int, default=10)
    parser.add_argument("--num-instances", type=int, default=30)
    parser.add_argument(
        "--sampling",
        type=str,
        default="hierarchical",
        choices=["unique", "uniform", "hierarchical"],
    )
    parser.add_argument(
        "--features-type",
        type=str,
        default="mnist_resnet18",
        choices=["onehot", "mnist_resnet18"],
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to pre-computed feature vectors. If the feature vectors have not been pre-computed,"
        "they will be extracted and saved to this path.",
    )
    parser.add_argument("--num-bags-train", type=int, default=2000)
    parser.add_argument("--num-bags-val", type=int, default=500)
    parser.add_argument("--num-bags-test", type=int, default=1000)
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--noise", type=float, default=0)

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=["attention_mil", "transmil", "additive_mil"],
    )
    parser.add_argument("--model-dims", type=int, default=20)
    parser.add_argument("--dropout", action="store_true")
    parser.add_argument("--n-out-layers", type=int, default=0)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--checkpoint", type=str, default="best")

    parser.add_argument("--explanation-methods", type=str, nargs="+", default=[None])
    parser.add_argument(
        "--evaluated-classes",
        type=str,
        default="all_classes",
        choices=["label_class", "predicted_class", "all_classes"],
    )
    parser.add_argument("--detach-pe", action="store_true")

    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    return args


def main():
    # Read args and create results directories
    args = get_args()
    print(json.dumps(vars(args), indent=4))
    timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    save_dir = os.path.join(args.results_dir, timestamp)
    os.makedirs(save_dir)
    print(f"Results will be written to: {save_dir}")
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    device = torch.device(args.device)

    # Load MNIST data and extract features if not already present
    if args.features_type == "mnist_resnet18":
        if not os.path.exists(args.features_path):
            os.makedirs(args.features_path)
        if not all(
            [f"class_{idx}.pt" in os.listdir(args.features_path) for idx in range(10)]
        ):
            print(
                f"No or not all MNIST features found at: {args.features_path}. Starting feature extraction."
            )
            mnist_features = get_MNIST_features(args.features_path, download=True)
            for idx in range(10):
                mnist_class_path = os.path.join(args.features_path, f"class_{idx}.pt")
                if not os.path.exists(mnist_class_path):
                    torch.save(mnist_features[idx], mnist_class_path)

    # Set up datasets
    dataset_train = get_number_mil_dataset(
        dataset_type=args.dataset_type,
        num_numbers=args.num_numbers,
        num_bags=args.num_bags_train,
        num_instances=args.num_instances,
        features_type=args.features_type,
        sampling=args.sampling,
        noise=args.noise,
        threshold=args.threshold,
        features_path=args.features_path,
    )
    dataset_val = get_number_mil_dataset(
        dataset_type=args.dataset_type,
        num_numbers=args.num_numbers,
        num_bags=args.num_bags_val,
        num_instances=args.num_instances,
        features_type=args.features_type,
        sampling=args.sampling,
        noise=args.noise,
        threshold=args.threshold,
        features_path=args.features_path,
    )
    dataset_test = get_number_mil_dataset(
        dataset_type=args.dataset_type,
        num_numbers=args.num_numbers,
        num_bags=args.num_bags_test,
        num_instances=args.num_instances,
        features_type=args.features_type,
        sampling=args.sampling,
        noise=args.noise,
        threshold=args.threshold,
        features_path=args.features_path,
    )
    collate_fn = (
        bag_collate_fn if args.model_type in ["attention_mil", "additive_mil"] else None
    )
    data_loader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, collate_fn=collate_fn
    )
    data_loader_val = DataLoader(
        dataset_val, batch_size=args.batch_size, collate_fn=collate_fn
    )
    data_loader_test = DataLoader(dataset_test, batch_size=1, collate_fn=collate_fn)

    all_results = []

    for repetition in range(args.num_repetitions):
        print(f"\n-- Repetition {repetition+1} --\n")

        model, classifier = get_model_and_classifier(
            model_type=args.model_type,
            num_features=dataset_train.num_features,
            num_classes=dataset_train.num_classes,
            model_dims=args.model_dims,
            dropout=args.dropout,
            n_out_layers=args.n_out_layers,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
        )

        train_metrics = train_model(
            classifier=classifier,
            num_classes=dataset_train.num_classes,
            data_loader_train=data_loader_train,
            data_loader_val=data_loader_val,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            warmup=args.warmup,
            tolerance=args.tolerance,
            patience=args.patience,
            checkpoint=args.checkpoint,
        )

        torch.save(model.state_dict(), os.path.join(save_dir, f"model_{repetition}.pt"))

        test_metrics = {}
        for explanation_type in args.explanation_methods:
            print(f"Evaluating {explanation_type}:")
            xmodel = get_xmodel(
                model_type=args.model_type,
                explanation_type=explanation_type,
                model=model,
                detach_pe=args.detach_pe,
            )
            all_labels, all_preds, scores = evaluate_explanation(
                xmodel=xmodel,
                classifier=classifier,
                data_loader_test=data_loader_test,
                explanation_type=explanation_type,
                evaluated_classes=args.evaluated_classes,
                ndgcn_n=None,
            )
            if dataset_train.num_classes == 2:
                all_preds = all_preds[:, -1]
            test_metrics["auc"] = roc_auc_score(
                all_labels.detach().cpu().numpy(),
                all_preds.detach().cpu().numpy(),
                multi_class="ovr",
            )
            test_metrics["bags"] = len(scores["auroc_pos"])
            test_metrics[explanation_type] = {
                key: val.mean().item() for key, val in scores.items()
            }
        all_metrics = {**train_metrics, "test": test_metrics}
        all_results.append(all_metrics)
        print(json.dumps(all_metrics, indent=4))

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=4)


if __name__ == "__main__":
    main()
