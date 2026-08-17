#!/bin/bash -l

n_folds=5
# all_folds=($(seq 0 $((n_folds-1))))
all_folds=( 0 )

datasets=( camelyon16 )
targets=( tumor_vs_normal )

#datasets=( tcga_luad camelyon16 tcga_hnsc)
#targets=( tp53 tumor_vs_normal hpv)

tcga_dir=/home/space/datasets/tcga
split_dir="$tcga_dir"/splits/xmil/classification

data_path_camelyon=/home/space/datasets/camelyon16
#data_path_luad="$tcga_dir"/luad
#data_path_hnsc="$tcga_dir"/hnsc
#data_paths=( "$data_path_luad" "$data_path_camelyon" )
data_paths=( "$data_path_camelyon" )

features_names=( virchow_v2_pt )

metadata_version=v001
num_epochs=1
warmup=0
results_root_dir=/home/mina/codes/my_codes/xMIL/results


for features_name in "${features_names[@]}"; do
  for ((i=0; i<${#datasets[@]}; i++)); do
    target=${targets[$i]}
    dataset=${datasets[$i]}
    data_path=${data_paths[$i]}

    split_name="$dataset"_"$target"_classification_cv"$n_folds"_0.csv
    split_path="$split_dir"/"$split_name"
    for test_fold in "${all_folds[@]}"; do

      val_fold=$(( (test_fold + 1) % "$n_folds" ))
      job_name="$test_fold"-"$features_name"-"$target"
      sbatch -J "$job_name" train_transmil_classification.sh \
      "$dataset" "$target" "$test_fold" "$metadata_version" "$split_path" \
      "$num_epochs" "$warmup" "$features_name" "$val_fold" "$results_root_dir" \
      "$data_path"

    done
  done
done
