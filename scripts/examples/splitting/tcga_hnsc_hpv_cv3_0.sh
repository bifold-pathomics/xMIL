#!/usr/bin/env bash


python3 split.py \
--metadata-paths /path/to/datasets/tcga/hnsc/metadata/v001/case_metadata.csv \
--target-path /path/to/datasets/tcga/splits/tcga_hnsc_hpv_cv3_0.csv \
--split-by case_id \
--target HPV_Status \
--strategy cross_validation \
--ratios '{"num_folds": 3}' \
--seed=0
