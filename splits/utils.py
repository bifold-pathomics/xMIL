from functools import partial

import pandas as pd

from splits.constants import (
    TCGA_NODE_STAGING_LABEL_MAPPING,
    TCGA_TUMOR_STAGING_LABEL_MAPPING,
    CAMELYON16_TUMOR_MAP,
    TCGA_HNSC_TSS_CV_LABEL_MAPPING,
    TCGA_HNSC_HPV_STATUS_MAP,
    TCGA_NSCLC_STUDY_MAP,
    RECIST_RESPONSE_BINARY,
    PATHWAYS,
    MUTATIONS,
    map_to_binary,
)


def get_label_mapping(target, label_threshold=None):
    """
    Transforms a target name into an actionable label mapping as defined in the constants.

    :param target: (str) The target identifier.
    :param label_threshold: (float) If the target is numeric, define a threshold for discretization.
    :return: (dict)
    """
    if target == "ajcc_pathologic_n":
        label_mapping = TCGA_NODE_STAGING_LABEL_MAPPING.get
    elif target == "ajcc_pathologic_t":
        label_mapping = TCGA_TUMOR_STAGING_LABEL_MAPPING.get
    elif target == "tss":
        label_mapping = TCGA_HNSC_TSS_CV_LABEL_MAPPING.get
    elif target == "HPV_Status":
        label_mapping = TCGA_HNSC_HPV_STATUS_MAP.get
    elif target == "type":
        label_mapping = CAMELYON16_TUMOR_MAP.get
    elif target == "study":
        label_mapping = TCGA_NSCLC_STUDY_MAP.get
    elif target == "recist_response":
        label_mapping = RECIST_RESPONSE_BINARY.get
    elif target in PATHWAYS + MUTATIONS:
        label_mapping = partial(map_to_binary, label_threshold)
    else:
        raise ValueError(f"No label mapping defined for target: {target}")
    return label_mapping


def balance_labels(
    data, label_cols, subsets="train", group_key=None, strategy="drop", seed=None
):
    # Transform subset arg into list format
    if subsets == "all":
        subsets = data["subset"].unique()
    elif isinstance(subsets, str):
        subsets = [subsets]

    # Set up data structures
    balanced_data = pd.DataFrame()
    unbalanced_data = data[~data["subset"].isin(subsets)]

    # Only balance labels when we are in unique label mode
    if len(label_cols) > 1:
        raise ValueError(
            "Cannot balance labels when there are multiple target columns. Please provide a single target."
        )
    elif len(label_cols) == 0:
        raise ValueError("No target column found. Please provide a target column.")
    else:
        lbl_col = label_cols[0]

    # Balance labels for each subset
    for subset in subsets:
        subset_data = data[data["subset"] == subset]
        if group_key is None:
            # Balance labels over the whole subset
            num_samples = subset_data[lbl_col].value_counts().min()
            for label in subset_data[lbl_col].unique():
                if strategy == "drop":
                    label_data = subset_data[subset_data[lbl_col] == label].sample(
                        n=num_samples, random_state=seed
                    )
                    balanced_data = pd.concat([balanced_data, label_data])
                else:
                    raise ValueError(f"Unknown label balancing strategy: {strategy}")
        else:
            # Balance labels over each group within the subset
            label_counts = (
                subset_data.groupby([group_key])
                .value_counts([lbl_col])
                .unstack()
                .fillna(0)
            )
            for group in label_counts.index:
                num_samples = int(label_counts.loc[group].min())
                if num_samples > 0:
                    for label in label_counts.columns:
                        if strategy == "drop":
                            label_data = subset_data[
                                (subset_data[lbl_col] == label)
                                & (subset_data[group_key] == group)
                            ].sample(n=num_samples, random_state=seed)
                            balanced_data = pd.concat([balanced_data, label_data])
                        else:
                            raise ValueError(
                                f"Unknown label balancing strategy: {strategy}"
                            )

    return pd.concat([balanced_data, unbalanced_data], ignore_index=True)
