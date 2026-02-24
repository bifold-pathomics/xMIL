from functools import partial

from splits.constants import (
    TCGA_HNSC_HPV_STATUS_MAP,
    TCGA_NSCLC_STUDY_MAP,
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
    if target == "HPV_Status":
        label_mapping = TCGA_HNSC_HPV_STATUS_MAP.get
    elif target == "study":
        label_mapping = TCGA_NSCLC_STUDY_MAP.get
    elif target in MUTATIONS:
        label_mapping = partial(map_to_binary, label_threshold)
    else:
        raise ValueError(f"No label mapping defined for target: {target}")
    return label_mapping
