#!/bin/bash -l

base_dir=/path/to/base/dir


# Attention MIL

python3 toy_experiment.py \
\
--results-dir ${base_dir}/results/four_bags/attnmil \
--num-repetitions 30 \
--dataset-type four_bags \
\
--num-numbers 10 \
--num-instances 30 \
--sampling hierarchical \
--features-type mnist_resnet18 \
--features-path ${base_dir}/data/mnist_resnet18 \
--num-bags-train 2000 \
--num-bags-val 500 \
--num-bags-test 1000 \
\
--model-type attention_mil \
--n-out-layers 1 \
--learning-rate 0.0001 \
--weight-decay 0.0 \
--num-epochs 1000 \
--patience 100 \
--checkpoint best \
\
--explanation-methods random attention perturbation_keep grad2 gi ig lrp \
--evaluated-classes all_classes


# TransMIL

python3 toy_experiment.py \
\
--results-dir ${base_dir}/results/pos_neg/transmil \
--num-repetitions 30 \
--dataset-type pos_neg \
\
--num-numbers 10 \
--num-instances 30 \
--sampling hierarchical \
--features-type mnist_resnet18 \
--features-path ${base_dir}/data/mnist_resnet18 \
--num-bags-train 2000 \
--num-bags-val 500 \
--num-bags-test 1000 \
\
--model-type transmil \
--n-out-layers 0 \
--learning-rate 0.0001 \
--weight-decay 0.0 \
--num-epochs 200 \
--patience 40 \
--checkpoint best \
\
--explanation-methods random attention perturbation_keep grad2 gi ig lrp \
--evaluated-classes all_classes


# Additive MIL

python3 toy_experiment.py \
\
--results-dir ${base_dir}/results/adjacent_pairs/addmil \
--num-repetitions 30 \
--dataset-type adjacent_pairs \
\
--num-numbers 10 \
--num-instances 30 \
--sampling hierarchical \
--features-type mnist_resnet18 \
--features-path ${base_dir}/data/mnist_resnet18 \
--num-bags-train 2000 \
--num-bags-val 500 \
--num-bags-test 1000 \
\
--model-type additive_mil \
--learning-rate 0.0001 \
--weight-decay 0.0 \
--num-epochs 1000 \
--patience 100 \
--checkpoint best \
\
--explanation-methods attention patch_scores \
--evaluated-classes all_classes
