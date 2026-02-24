from torch.utils.data import DataLoader

from datasets.mil import MILSlideDataset


class DatasetFactory:

    @staticmethod
    def build(dataset_args, model_args):

        if model_args["aggregation_model"] == "transmil":
            collate_fn = None
        else:
            collate_fn = MILSlideDataset.bag_collate_fn

        if dataset_args.get("train_subsets") is not None:
            train_dataset = DatasetFactory._build_image_dataset(dataset_args, "train")
            train_loader = DataLoader(
                train_dataset,
                batch_size=dataset_args["train_batch_size"],
                shuffle=True,
                collate_fn=collate_fn,
            )
        else:
            train_dataset, train_loader = None, None

        if dataset_args.get("val_subsets") is not None:
            val_dataset = DatasetFactory._build_image_dataset(dataset_args, "val")
            val_loader = DataLoader(
                val_dataset,
                batch_size=dataset_args["val_batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
            )
        else:
            val_dataset, val_loader = None, None

        if dataset_args.get("test_subsets") is not None:
            test_dataset = DatasetFactory._build_image_dataset(dataset_args, "test")
            test_loader = DataLoader(
                test_dataset,
                batch_size=dataset_args["val_batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
            )
        else:
            test_dataset, test_loader = None, None

        return (
            train_dataset,
            train_loader,
            val_dataset,
            val_loader,
            test_dataset,
            test_loader,
        )

    @staticmethod
    def _build_image_dataset(args, stage):

        if stage == "train":
            subsets = args["train_subsets"]
            bag_size = args.get("train_bag_size", None)
        elif stage == "val":
            subsets = args["val_subsets"]
            bag_size = None
        elif stage == "test":
            subsets = args["test_subsets"]
            bag_size = None
        else:
            raise ValueError(f"Unknown stage: {stage}")

        dataset = MILSlideDataset(
            split_path=args["split_path"],
            metadata_dirs=args["metadata_dirs"],
            subsets=subsets,
            patches_dirs=args["patches_dirs"],
            features_dirs=args["features_dirs"],
            label_cols=args.get("targets", ["label"]),
            bag_size=bag_size,
            patch_filters=args.get("patch_filters", None),
            preload_features=args.get("preload_data", False),
            drop_duplicates=args.get("drop_duplicates", "sample"),
            max_bag_size=args.get("max_bag_size", None),
        )

        return dataset
