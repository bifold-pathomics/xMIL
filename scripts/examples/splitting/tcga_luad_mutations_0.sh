#!/usr/bin/env bash


python3 split.py \
--metadata-paths /path/to/datasets/tcga/luad/metadata/v002/case_metadata.csv \
--target-path /path/to/datasets/tcga/splits/tcga_luad_tp53_60_15_25_0.csv \
--split-by case_id \
--target TP53 \
--strategy train_val_test \
--ratios '{"train": 0.6, "val": 0.15, "test": 0.25}' \
--seed=0
