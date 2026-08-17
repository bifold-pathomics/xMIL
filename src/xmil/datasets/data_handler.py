import os
import json

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import ast


class MetadataHandler:

    @staticmethod
    def load_split_metadata(
        split_path,
        metadata_dirs,
        subsets,
        label_cols,
        modalities=None,
        merge_on="case_id",
    ):
        """
        Loads and fuses metadata and split information, filtered by the given subsets.

        :param split_path: (str)
        :param metadata_dirs: (list<str>)
        :param subsets: (list<str>)
        :param label_cols: (list<str>)
        :param modalities: (list<str>)
        :param merge_on: (str)
        :return: (pandas.DataFrame)
        """
        # Read split
        split = pd.read_csv(split_path)
        split["subset"] = split["subset"].astype(
            str
        )  # for compatibility with 'subsets' arg
        split = split[split["subset"].isin(subsets)][[merge_on, "subset"] + label_cols]
        # Read and merge case metadata
        case_metadata = pd.DataFrame()
        for idx, metadata_dir in enumerate(metadata_dirs):
            case_met_ = pd.read_csv(os.path.join(metadata_dir, "case_metadata.csv"))
            case_met_.insert(0, "source_id", idx)
            case_metadata = pd.concat(
                [case_metadata, case_met_], axis=0, ignore_index=True
            )
        split = split.merge(
            case_metadata, how="inner", on=merge_on, suffixes=(None, "_ctrl")
        )
        # Sanity check the given data sources
        if "source_id_ctrl" in split:
            if split["source_id"].equals(split["source_id_ctrl"]):
                split = split.drop("source_id_ctrl", axis=1)
            else:
                raise ValueError(
                    "Given data sources do not match the data sources of the split"
                )
        # Read and merge modality-specific metadata
        if modalities is not None:
            for modality in modalities:
                modality_metadata = pd.DataFrame()
                for idx, metadata_dir in enumerate(metadata_dirs):
                    mod_met_ = pd.read_csv(
                        os.path.join(metadata_dir, f"{modality}_metadata.csv")
                    )
                    modality_metadata = pd.concat(
                        [modality_metadata, mod_met_], axis=0, ignore_index=True
                    )
                split = split.merge(
                    modality_metadata,
                    how="inner",
                    on=merge_on,
                    suffixes=(None, "_ctrl"),
                )
        return split


class SlideDataHandler:

    @staticmethod
    def load_patch_metadata(slide_metadata, patches_dirs, patch_filters):
        """
        Loads patch metadata from the patch-level metadata frame for all given slides. Filters the patches according
        to the given filters.

        :param slide_metadata: (pd.DataFrame) DataFrame containing slide ids to load patches for.
            Columns required: 'slide_id' identifying where to find the slide; 'source_id' indexing patches_dirs.
        :param patches_dirs: (list<str>) Directories where the patch metadata are stored.
        :param patch_filters: (dict)
            - 'has_annot' (list) Only use patches with some annotation of one of the given classes.
            - 'exclude_annot': (list) Only use patches with no annotation of the given classes.
        :return: (list, list) Per slide: tensor of indices to filter the feature vectors per slide, tensor of patch ids.
            Ordered by the given slide barcodes.
        """
        print("Loading patch metadata")
        feature_indices, patch_ids, positions_x, positions_y = [], [], [], []
        for idx, row in tqdm(slide_metadata.iterrows(), total=len(slide_metadata)):
            source_id, slide_id = row[["source_id", "slide_id"]]
            slide_patches = pd.read_csv(
                os.path.join(patches_dirs[source_id], slide_id, "metadata/df.csv"),
                index_col=0,
            )
            slide_patches = slide_patches.sort_values("patch_id")
            slide_patches.insert(0, "slide_name", slide_id)
            slide_patches.insert(1, "feature_idx", range(len(slide_patches)))
            slide_patches = SlideDataHandler.filter_patches(
                slide_patches, patch_filters
            )
            feature_indices.append(torch.tensor(slide_patches["feature_idx"].values))
            patch_ids.append(torch.tensor(slide_patches["patch_id"].values))
            positions_x.append(
                torch.tensor(
                    slide_patches["position_rel"]
                    .apply(lambda x: ast.literal_eval(x)[0])
                    .values
                )
            )
            positions_y.append(
                torch.tensor(
                    slide_patches["position_rel"]
                    .apply(lambda x: ast.literal_eval(x)[1])
                    .values
                )
            )
        return feature_indices, patch_ids, positions_x, positions_y

    @staticmethod
    def filter_patches(patch_metadata, patch_filters):
        """
        Filters the patches according to the given filters.

        :param patch_metadata: (pandas.DataFrame) Patch-level metadata, esp. annotation_{idx} columns.
        :param patch_filters: (dict)
            - 'has_annot' (list) Only use patches with some annotation of one of the given classes.
            - 'exclude_annot': (list) Only use patches with no annotation of the given classes.
        :return: (pandas.DataFrame)
        """
        if patch_filters is not None:
            annot_classes = patch_metadata["annotation_classes"].apply(
                lambda x: pd.Series(json.loads(x))
            )
            if "has_annot" in patch_filters:
                annot_cols = [
                    str(annot_cls) for annot_cls in patch_filters["has_annot"]
                ]
                patch_metadata = patch_metadata[
                    annot_classes[annot_cols].sum(axis=1) > 0
                ]
            if "exclude_annot" in patch_filters:
                annot_cols = [
                    str(annot_cls) for annot_cls in patch_filters["exclude_annot"]
                ]
                patch_metadata = patch_metadata[
                    annot_classes[annot_cols].sum(axis=1) == 0
                ]
        return patch_metadata

    @staticmethod
    def load_features(path, feature_indices=None, patch_ids=None):
        """
        Loads features of one slide into RAM. Pass feature_indices or patch ids to only load selected features.

        :param path: (str) Directory where the patch features are stored.
        :param feature_indices: (torch.Tensor) Selected features indices to load (for features saved in list-style).
        :param patch_ids: (torch.Tensor) Selected patch IDs to load (for features saved in dict-style).
        :return: (torch.Tensor) The feature vectors of shape (num_patches, num_features).
        """
        if os.path.exists(f"{path}.npz"):
            features = dict(np.load(f"{path}.npz"))
            keys, features = list(zip(*list(features.items())))
            keys = torch.tensor(list(map(int, list(keys))))
            keys_argsort = torch.argsort(keys)
            features = torch.from_numpy(np.stack(features)).squeeze()
            features = features[keys_argsort]
            if patch_ids is not None:
                patch_ids_to_idx = {
                    patch_id: idx
                    for patch_id, idx in zip(keys.tolist(), keys_argsort.tolist())
                }
                vec_idx = list(map(patch_ids_to_idx.get, patch_ids.tolist()))
                features = features[vec_idx]
        elif os.path.exists(f"{path}.pt"):  # TODO implement sorting by json file
            features = torch.load(f"{path}.pt").squeeze()
            if feature_indices is not None:
                features = features[feature_indices]
        else:
            feature_idx_list = sorted(
                [
                    int(os.path.splitext(file)[0])
                    for file in os.listdir(path)
                    if file.endswith(".pt")
                ]
            )
            features = torch.stack(
                [
                    torch.load(os.path.join(path, f"{idx}.pt"))
                    for idx in feature_idx_list
                ]
            ).squeeze()
            if feature_indices is not None:
                features = features[feature_indices]
        return features.float()

    @staticmethod
    def sample_features(
        features,
        bag_size,
        sort_sampled_patches=False,
        patch_ids=None,
        positions_x=None,
        positions_y=None,
    ):
        """
        Randomly samples a number of features from the given feature tensor. Pass patch_ids to recover the identity of
        the sampled features.

        :param features: (torch.Tensor) Feature vectors of shape (num_patches, num_features).
        :param bag_size: (int) Number of features to sample. If this is smaller than the number of feature vectors,
            zero padding will be applied.
        :param patch_ids: (torch.Tensor) List of patch IDs corresponding to the feature vectors.
        :return:
            - features: (torch.Tensor) Sampled feature vectors of shape (bag_size, num_features).
            - patch_ids: (torch.Tensor) Sampled patch IDs of shape (bag_size).
        """
        idx_list = torch.randperm(features.shape[0])[:bag_size]

        if sort_sampled_patches:
            idx_list, _ = torch.sort(idx_list)

        features = features[idx_list]
        if patch_ids is not None:
            patch_ids = patch_ids[idx_list]
        if positions_x is not None:
            positions_x = positions_x[idx_list]
        if positions_y is not None:
            positions_y = positions_y[idx_list]
        if len(features) < bag_size:
            features = torch.cat(
                (features, torch.zeros(bag_size - features.shape[0], features.shape[1]))
            )
            if patch_ids is not None:
                patch_ids = torch.cat(
                    (patch_ids, torch.full((bag_size - len(patch_ids),), -1))
                )
            if positions_x is not None:
                positions_x = torch.cat(
                    (positions_x, torch.full((bag_size - len(positions_x),), -1))
                )
            if positions_y is not None:
                positions_y = torch.cat(
                    (positions_y, torch.full((bag_size - len(positions_y),), -1))
                )
        return features, patch_ids, positions_x, positions_y


class OmicsDataHandler:

    @staticmethod
    def load_feature_metadata(features_dirs):
        feature_metadata = {}
        for source_id, features_dir in enumerate(features_dirs):
            with open(os.path.join(features_dir, f"metadata.json")) as f:
                feature_metadata[source_id] = json.load(f)
        return feature_metadata

    @staticmethod
    def load_features(path):
        """
        Loads features of one slide into RAM. Pass feature_indices to only load selected features.

        :param path: (str) Directory where the patch features are stored.
        :return: (torch.Tensor) The feature vectors of shape (num_bags, num_features).
        """
        if os.path.exists(f"{path}.pt"):
            features = torch.load(f"{path}.pt", weights_only=True).squeeze()
        elif os.path.exists(f"{path}.npy"):
            features = torch.tensor(np.load(f"{path}.npy")).squeeze()
        else:
            raise ValueError(f"Couldn't find omics features file: {path}.pt")
        return features.float()


class SurvivalDiscretizer:
    """Discretize the survival values based on the quartiles."""

    def __init__(self, survival_col_name="survival_continuous"):
        """
        Initialize SurvivalDiscretizer
        :param survival_col_name: the name of the column containing the survival values.
        """
        self.survival_col = survival_col_name

    def compute_apply_discretizer(self, dataframe):
        """
        Compute the discretizing bins and also apply it to the given data frame.

        :param dataframe: (DataFrame) the dataframe with the survival values in self.survival_col.
        :return:
        """
        # todo: we can add q (# of bins) as the input to have more flexibility.
        dataframe["survival_group"], discretizer_bins = pd.qcut(
            dataframe[self.survival_col], q=4, labels=[0, 1, 2, 3], retbins=True
        )

        # in order to make sure that the least survival is 0 and there is no maximum for it:
        discretizer_bins[0] = 0
        discretizer_bins[-1] = np.inf

        return dataframe, discretizer_bins

    def apply_discretizer(self, dataframe, bins):
        """
        Use the given bins for discretizing the survival values.
        :param dataframe: (DataFrame) the dataframe with the survival values in self.survival_col
        :param bins: (numpy array) the array containing teh bins for the descritization task.
        :return:
        """
        dataframe["survival_group"] = pd.cut(
            dataframe[self.survival_col],
            bins=bins,
            labels=[0, 1, 2, 3],
            include_lowest=True,
        )
        return dataframe
