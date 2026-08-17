#!/bin/bash -l

#SBATCH -a 0
#SBATCH --partition=gpu-test
#SBATCH --exclude=head022,head076,head077,head073
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=128GB

#SBATCH -o /home/mina/codes/my_codes/xMIL/logs/%A_%a.txt
#SBATCH -D /home/mina/codes/my_codes/xMIL
# ------ SPECIFY ------------------
# source /home/mina/miniconda3/bin/activate
# conda activate xmil
# ------------------------

dataset=$1
target=$2
test_fold=$3
metadata_version=$4
split_path=$5
num_epochs=$6
warmup=$7
features_name=$8
val_fold=$9
results_root_dir=${10}
data_path=${11}

device=cuda
model=transmil

# ------ SPECIFY ------------------
results_dir="$results_root_dir"/"$dataset"/$target/$model
mkdir -p "$results_dir"
# ------------------------
#bag_sizes=( 2048 )
#batch_sizes=( 5 )
#dropouts_att=( 0 0.5 )
#dropouts_class=( 0 0.5 )
#dropouts_feat=( 0 0.2 )
#learning_rates=( 0.0002 0.002 )
#weight_decays=( 0 0.01 )
#grad_clip=( 1 )
#seeds=( 0 )

bag_sizes=( 2048 )
batch_sizes=( 5 )
dropouts_att=( 0 0.5 )
dropouts_class=( 0 0.5 )
dropouts_feat=( 0 0.2 )
learning_rates=( 2e-5 2e-4 2e-3 )
weight_decays=( 0 0.01 )
grad_clip=( -1 1 )
seeds=( 0 )

# define the train, val, test subsets
n_folds=5
all_folds=($(seq 0 $((n_folds-1))))

val_folds=( "$val_fold" )
test_folds=( "$test_fold" )

test_val_folds=( "$val_fold" "$test_fold" )

train_folds=()
for element in "${all_folds[@]}"; do
    if [[ ! " ${test_val_folds[@]} " =~ " $element " ]]; then
      train_folds+=("$element")
    fi
done

train_folds_json="[[$(printf "\"%s\"," "${train_folds[@]}" | sed 's/,$//')]]"
test_folds_json="[[$(printf "\"%s\"," "${test_folds[@]}" | sed 's/,$//')]]"
val_folds_json="[[$(printf "\"%s\"," "${val_folds[@]}" | sed 's/,$//')]]"

patches_dir=patches/20x
features_dir=features/20x/"$features_name"


if [ "$features_name" = "uni2_pt" ]; then
    input_dim=1536
elif [ "$features_name" = "uni_pt" ]; then
    input_dim=1024
elif [ "$features_name" = "virchow_v2_pt" ]; then
    input_dim=1280
fi

apptainer exec --nv \
--bind "$PWD:$PWD" \
--pwd "$PWD" \
--bind /home/space/datasets/camelyon16:/home/space/datasets/camelyon16 \
--bind /home/space/datasets/tcga:/home/space/datasets/tcga \
--bind /home/space/pathomics:/home/space/pathomics \
/home/space/pathomics/containers/mamba_container.sif \
env PYTHONPATH="$PWD/src" \
python3 scripts/train_array.py \
\
--split-path "$split_path" \
--metadata-dirs "$data_path"/metadata/"$metadata_version" \
--patches-dirs "$data_path"/"$patches_dir" \
--features-dirs "$data_path"/"$features_dir" \
--results-dir "$results_dir" \
\
--warmup "$warmup" \
\
--train-subsets "$train_folds_json" \
--val-subsets "$val_folds_json" \
--test-subsets "$test_folds_json" \
--drop-duplicates sample \
--train-bag-size ${bag_sizes[@]} \
--stop-criterion perf_metric \
--preload-data \
\
--aggregation-model $model \
--input-dim $input_dim \
--head-dim 2 \
--head-type classification \
--features-dim 256 \
--dropout-att ${dropouts_att[@]} \
--dropout-class ${dropouts_class[@]} \
--dropout-feat ${dropouts_feat[@]} \
--n-layers 2 \
--pool-method cls_token \
--no-ppeg \
--no-attn-residual \
\
--train-batch-size ${batch_sizes[@]} \
--val-batch-size 1 \
--learning-rate ${learning_rates[@]} \
--weight-decay ${weight_decays[@]} \
--loss-type cross-entropy \
--num-epochs $num_epochs \
--val-interval 1 \
--grad-clip ${grad_clip[@]} \
\
--test-checkpoint best \
\
--seed ${seeds[@]} \
\
--device $device \
--save-folder hashlib_sha256 \
--num-workers 4
#--max-bag-size $max_bag_size
