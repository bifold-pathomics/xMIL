#!/bin/bash

approach=drop

device=cuda
max_bag_size=15000


flip_perc=1
strategy="$flip_perc"%-of-all

model_path=/path/to/model_path

results_dir=/path/to/results_path
mkdir "$results_dir"


explanation_folder=$model_path/explanations
explanation_path=$explanation_folder/test_predictions.csv

python3 evaluation_patch_flipping.py \
\
--model-path=$model_path \
--results-dir="$results_dir" \
--sel-checkpoint=best \
--dataset=test \
--explanation-types lrp perturbation_keep attention gi grad2 ig \
--explain-scores-path=$explanation_path \
--precomputed-heatmap-types lrp perturbation_keep attention gi grad2 ig \
--max-bag-size=$max_bag_size \
--strategy=$strategy \
--explained-rel=logit \
--lrp-params '{"gamma": 0, "eps": 1e-8, "no_bias": 1}' \
--approach=$approach \
--device=$device \
--flipping \
--baseline \
--morl-abs

