from torch.utils.data import DataLoader

from datasets.mil import MILSlideDataset, MILOmicsDataset
from datasets.samplers import sampling_args_factory


class DatasetFactory:

    @staticmethod
    def build(dataset_args, model_args):

        if model_args.get("mode", "image-only") == "image-only":
            dataset_collate_fn = MILSlideDataset.bag_collate_fn
            dataset_build_fn = DatasetFactory._build_image_dataset
        elif model_args["mode"] == "omics-only":
            dataset_collate_fn = MILOmicsDataset.bag_collate_fn
            dataset_build_fn = DatasetFactory._build_omics_dataset
        else:
            raise ValueError(f"Unknown mode: {model_args['mode']}")

        if (
            model_args["aggregation_model"] == "transmil"
            or model_args["aggregation_model"] == "mamba_mil"
        ):
            collate_fn = None
        else:
            collate_fn = dataset_collate_fn

        if dataset_args.get("train_subsets") is not None:
            train_dataset = dataset_build_fn(dataset_args, "train")
            dataset_args["survival_bins"] = train_dataset.survival_bins
            train_loader = DataLoader(
                train_dataset,
                batch_size=dataset_args["train_batch_size"],
                **sampling_args_factory(dataset_args["sampler"], train_dataset),
                collate_fn=collate_fn,
                num_workers=dataset_args.get("num_workers", 0),
            )

        else:
            train_dataset, train_loader = None, None

        if dataset_args.get("val_subsets") is not None:
            val_dataset = dataset_build_fn(dataset_args, "val")
            val_loader = DataLoader(
                val_dataset,
                batch_size=dataset_args["val_batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=dataset_args.get("num_workers", 0),
            )
        else:
            val_dataset, val_loader = None, None

        if dataset_args.get("test_subsets") is not None:
            test_dataset = dataset_build_fn(dataset_args, "test")
            test_loader = DataLoader(
                test_dataset,
                batch_size=dataset_args["val_batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=dataset_args.get("num_workers", 0),
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
            num_repetitions = 1
        elif stage == "val":
            subsets = args["val_subsets"]
            bag_size = (
                args.get("train_bag_size", None)
                if args.get("test_bag_sampling", False)
                else None
            )
            num_repetitions = args.get("test_repetitions", 1)
        elif stage == "test":
            subsets = args["test_subsets"]
            bag_size = (
                args.get("train_bag_size", None)
                if args.get("test_bag_sampling", False)
                else None
            )
            num_repetitions = args.get("test_repetitions", 1)
        else:
            raise ValueError(f"Unknown stage: {stage}")

        subsets = [
            str(i) for i in subsets
        ]  # hotfix, fix properly later (e.g. at the input level)

        dataset = MILSlideDataset(
            split_path=args["split_path"],
            metadata_dirs=args["metadata_dirs"],
            subsets=subsets,
            patches_dirs=args["patches_dirs"],
            features_dirs=args["features_dirs"],
            label_cols=args.get("targets", ["label"]),
            bag_size=bag_size,
            sort_sampled_patches=args.get("sort_sampled_patches", False),
            num_repetitions=num_repetitions,
            patch_filters=args.get("patch_filters", None),
            preload_features=args.get("preload_data", False),
            drop_duplicates=args.get("drop_duplicates", "sample"),
            max_bag_size=args.get("max_bag_size", None),
            min_bag_size=args.get("min_bag_size", 0),
            survival=(args.get("head_type", "classification") == "survival"),
            survival_bins=args.get("survival_bins", None),
        )

        return dataset

    @staticmethod
    def _build_omics_dataset(args, stage):
        # todo: survival discretizer must be added to MILOmicsDataset and then to this method

        if stage == "train":
            subsets = args["train_subsets"]
        elif stage == "val":
            subsets = args["val_subsets"]
        elif stage == "test":
            subsets = args["test_subsets"]
        else:
            raise ValueError(f"Unknown stage: {stage}")

        dataset = MILOmicsDataset(
            split_path=args["split_path"],
            metadata_dirs=args["metadata_dirs"],
            subsets=subsets,
            features_dirs=args["features_dirs"],
            label_cols=args.get("targets", ["label"]),
            preload_features=args.get("preload_data", False),
            drop_duplicates=args.get("drop_duplicates", "sample"),
        )

        return dataset
