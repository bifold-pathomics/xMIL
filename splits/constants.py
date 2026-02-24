TCGA_HNSC_HPV_STATUS_MAP = {"HNSC_HPV-": 0, "HNSC_HPV+": 1}

TCGA_NSCLC_STUDY_MAP = {"TCGA LUAD": 0, "TCGA LUSC": 1}

MUTATIONS = ["TP53"]


def map_to_binary(threshold, decimal_value):
    if decimal_value >= threshold:
        return 1
    else:
        return 0
