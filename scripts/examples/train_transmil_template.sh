#!/bin/bash -l

bag_size=2048
batch_size=5
dropouts_att=0.5
dropouts_class=0.5
dropouts_feat=0.2
learning_rate=0.0002
weight_decay=0.0
max_bag_size=24000


python3 train.py \
\
--split-path /path/to/tcga/splits/tcga_nsclc_study_60_15_25_0.csv \
--metadata-dirs /path/to/tcga/luad/metadata/v001 /path/to/tcga/lusc/metadata/v001 \
--patches-dirs /path/to/tcga/luad/patches/20x /path/to/tcga/lusc/patches/20x \
--features-dirs /path/to/tcga/luad/features/20x/ctranspath_pt /path/to/tcga/lusc/features/20x/ctranspath_pt \
--results-dir /path/to/results/nsclc/transmil \
\
--train-subsets train \
--val-subsets val \
--test-subsets test \
--drop-duplicates sample \
--train-bag-size $bag_size \
--max-bag-size $max_bag_size \
--preload-data \
\
--aggregation-model transmil \
--input-dim 768 \
--num-classes 2 \
--num-features 256 \
--dropout-att $dropouts_att \
--dropout-class $dropouts_class \
--dropout-feat $dropouts_feat \
--n-layers 2 \
--pool-method cls_token \
\
--train-batch-size $batch_size \
--val-batch-size 1 \
--learning-rate $learning_rate \
--weight-decay $weight_decay \
--objective cross-entropy \
--num-epochs 200 \
--val-interval 1 \
--stop-criterion auc \
\
--test-checkpoint best \
\
--device cuda
