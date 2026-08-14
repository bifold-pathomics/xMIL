#!/bin/bash -l

bag_size=2048
batch_size=32
dropout=0
learning_rate=0.002
weight_decay=0.0
max_bag_size=24000


python3 train.py \
\
--split-path /path/to/tcga/splits/tcga_nsclc_study_60_15_25_0.csv \
--metadata-dirs /path/to/tcga/luad/metadata/v001 /path/to/tcga/lusc/metadata/v001 \
--patches-dirs /path/to/tcga/luad/patches/20x /path/to/tcga/lusc/patches/20x \
--features-dirs /path/to/tcga/luad/features/20x/ctranspath_pt /path/to/tcga/lusc/features/20x/ctranspath_pt \
--results-dir /path/to/results/nsclc/attnmil \
\
--train-subsets train \
--val-subsets val \
--test-subsets test \
--drop-duplicates sample \
--train-bag-size $bag_size \
--max-bag-size $max_bag_size \
--preload-data \
\
--aggregation-model attention_mil \
--input-dim 768 \
--num-classes 2 \
--features-dim 256 \
--inner-attention-dim 128 \
--dropout $dropout \
--dropout-strategy all \
--n-out-layers=0 \
\
--train-batch-size $batch_size \
--val-batch-size 1 \
--learning-rate $learning_rate \
--weight-decay $weight_decay \
--objective cross-entropy \
--num-epochs 1000 \
--val-interval 1 \
--stop-criterion auc \
\
--test-checkpoint best \
\
--device cuda
