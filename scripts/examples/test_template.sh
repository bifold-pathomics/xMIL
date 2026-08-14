#!/bin/bash -l


python3 test.py \
\
--model-dir /path/to/trained_model_dir \
--test-checkpoint best \
--results-dir /path/to/results_dir \
\
--test-subsets test \
--drop-duplicates sample \
--val-batch-size 1 \
--max-bag-size 24000 \
\
--explanation-types lrp gi grad2 attention \
--explained-rel logit \
\
--device cpu
