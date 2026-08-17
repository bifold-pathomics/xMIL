TCGA_NODE_STAGING_LABEL_MAPPING = {
    "N0": 0,
    "N1": 1,
    "N2": 1,
    "N2a": 1,
    "N2b": 1,
    "N2c": 1,
    "N3": 1,
}

TCGA_TUMOR_STAGING_LABEL_MAPPING = {
    "T1": 0,
    "T2": 0,
    "T3": 1,
    "T4": 1,
    "T4a": 1,
    "T4b": 1,
}

TCGA_HNSC_TSS_CV_LABEL_MAPPING = {
    "4P": 0,
    "BA": 0,
    "BB": 0,
    "C9": 0,
    "CN": 0,
    "CQ": 0,
    "CV": 1,
    "CX": 0,
    "D6": 0,
    "DQ": 0,
    "F7": 0,
    "H7": 0,
    "HD": 0,
    "HL": 0,
    "IQ": 0,
    "KU": 0,
    "MT": 0,
    "MZ": 0,
    "P3": 0,
    "QK": 0,
    "RS": 0,
    "T2": 0,
    "T3": 0,
    "TN": 0,
    "UF": 0,
    "WA": 0,
}

TCGA_HNSC_HPV_STATUS_MAP = {"HNSC_HPV-": 0, "HNSC_HPV+": 1}

TCGA_NSCLC_STUDY_MAP = {"TCGA LUAD": 0, "TCGA LUSC": 1}

CAMELYON16_TUMOR_MAP = {"normal": 0, "tumor": 1}

RECIST_RESPONSE_BINARY = {
    "PD": 0,
    "SD": 0,
    "PR": 1,
    "CR": 1,
}

PATHWAYS = [
    "Androgen",
    "EGFR",
    "Estrogen",
    "Hypoxia",
    "JAK-STAT",
    "MAPK",
    "NFkB",
    "p53",
    "PI3K",
    "TGFb",
    "TNFa",
    "Trail",
    "VEGF",
    "WNT",
]

MUTATIONS = ["EGFR", "ALK", "KRAS", "TP53"]


def map_to_binary(threshold, decimal_value):
    if decimal_value >= threshold:
        return 1
    else:
        return 0
