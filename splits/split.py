import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GroupShuffleSplit,
    KFold,
    GroupKFold,
)

from splits.utils import balance_labels


def split(
    metadata,
    split_by,
    target=None,
    label_mapping=None,
    groups=None,
    strategy="train_val_test",
    ratios=None,
    balance_subsets=None,
    balance_group=None,
    seed=0,
    survival=False,
    continuous_target=False,
    censorship_mapping=None,
):
    """
    :param metadata: (pd.DataFrame)
    :param split_by: (str) Column of the data frame to split by. The splitting will be done on unique values.
    :param target: (list | str) column of the data frame to derive the label from. if for survival, it must be a list
    with two elements, the first one is the column name for the overall survival and second one is for the censorship,
    i.e., target=[<survival_column_name>, <censorship_column_name>]
    :param label_mapping: (dict) Mapping from values in the target column of the metadata frame to integer labels.
        Rows with values not represented in this dict will be removed.
    :param groups: (str) Column of the data frame to group data by. If provided, all data of the same group will be in
        the same subset.
    :param strategy: (str) How to split, e.g., 'train_test', 'train_val_test', 'cross_validation'
    :param ratios: (dict) Dict describing the split ratios, e.g., {'train': 0.6, 'val': 0.1, 'test': 0.3} for strategy
        'train_val_test', or {'num_folds': 5} for strategy 'cross_validation'
    :param balance_subsets: (str) Subsets in which to enforce the same amount of number of samples per label. E.g.,
        'train' for training set, 'all' for every subset. If None, no label balancing will be performed.
    :param balance_group: (str) A key of the metadata table. If provided, labels will be balanced across every
        value of this key (e.g., 'tss'). If None, labels are balanced across the whole subset.
    :param seed: (int)
    :param survival: (bool) whether the split is for a survival analysis
    :param continuous_target: (bool) whether the split is for a continuous variable
    :param censorship_mapping: (dict) the censorship mapping
    :return: (pd.DataFrame) A reduced data frame with columns for ids (see split_by), target and label, subset.
    """
    if survival:  # make sure that for survival the continuous_target is turned on
        continuous_target = True
    # Filter metadata and create labels
    metadata = metadata.drop_duplicates(subset=[split_by])
    if isinstance(target, list):
        for t in target:
            metadata = metadata[~metadata[t].isna()]
    elif isinstance(target, str):
        metadata = metadata[~metadata[target].isna()]
    if survival:
        if not isinstance(target, list) and len(target) != 2:
            raise ValueError(
                "targets should be a list of two strings for generating a split appropriate",
                "for a survival analysis",
            )
        metadata = metadata.rename(columns={target[0]: "survival_continuous"})

        censorship_mapping = (
            {"0:LIVING": 1, "1:DECEASED": 0}
            if censorship_mapping is None
            else censorship_mapping
        )
        metadata["censorship"] = metadata[target[1]].map(censorship_mapping)

        target = ["survival_continuous", "censorship"]
    elif target is not None and label_mapping is not None:
        metadata.insert(
            len(metadata.columns), "label", metadata[target].apply(label_mapping)
        )
        metadata = metadata.dropna(subset="label")
        metadata["label"] = metadata["label"].astype(int)

    # Split data
    if strategy == "train_test":
        if groups is None:
            train_set, test_set = train_test_split(
                metadata, test_size=ratios["test"], random_state=seed
            )
        else:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=ratios["test"], random_state=seed
            )
            train_idx, test_idx = next(
                splitter.split(metadata, groups=metadata[groups])
            )
            train_set, test_set = metadata.iloc[train_idx], metadata.iloc[test_idx]
        train_set.insert(len(train_set.columns), "subset", "train")
        test_set.insert(len(test_set.columns), "subset", "test")
        split_df = pd.concat([train_set, test_set], axis=0, ignore_index=True)
    elif strategy == "train_val_test":
        if groups is None:
            train_val_set, test_set = train_test_split(
                metadata, test_size=ratios["test"], random_state=seed
            )
            sub_val_ratio = ratios["val"] / (ratios["train"] + ratios["val"])
            train_set, val_set = train_test_split(
                train_val_set, test_size=sub_val_ratio, random_state=seed
            )
        else:
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=ratios["test"], random_state=seed
            )
            train_val_idx, test_idx = next(
                splitter.split(metadata, groups=metadata[groups])
            )
            train_val_set, test_set = (
                metadata.iloc[train_val_idx],
                metadata.iloc[test_idx],
            )
            sub_val_ratio = ratios["val"] / (ratios["train"] + ratios["val"])
            splitter = GroupShuffleSplit(
                n_splits=1, test_size=sub_val_ratio, random_state=seed
            )
            train_idx, val_idx = next(
                splitter.split(train_val_set, groups=train_val_set[groups])
            )
            train_set, val_set = (
                train_val_set.iloc[train_idx],
                train_val_set.iloc[val_idx],
            )
        train_set.insert(len(train_set.columns), "subset", "train")
        val_set.insert(len(val_set.columns), "subset", "val")
        test_set.insert(len(test_set.columns), "subset", "test")
        split_df = pd.concat([train_set, val_set, test_set], axis=0, ignore_index=True)
    elif strategy == "cross_validation":
        folds = []
        if groups is None:
            splitter = KFold(
                n_splits=ratios["num_folds"], shuffle=True, random_state=seed
            )
            for _, test_idx in splitter.split(metadata):
                folds.append(metadata.iloc[test_idx])
        else:
            splitter = GroupKFold(n_splits=ratios["num_folds"])
            for _, test_idx in splitter.split(metadata, groups=metadata[groups]):
                folds.append(metadata.iloc[test_idx])
        for idx, fold in enumerate(folds):
            fold.insert(len(fold.columns), "subset", idx)
        split_df = pd.concat(folds, axis=0, ignore_index=True)
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")

    # Label balancing
    if balance_subsets is not None:
        split_df = balance_labels(
            data=split_df, subsets=balance_subsets, group_key=balance_group, seed=seed
        )

    if continuous_target:
        return split_df[[split_by] + target + ["subset"]]
    else:
        return split_df[[split_by, target, "label", "subset"]]
