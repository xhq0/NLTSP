#!/bin/bash

make_dataset_script="nltsp_make_dataset_test.py"
json_files_path="dataset/measure_records/k80/"          # Be sure to end with /
output_path="pkl_dataset/"                              # Be sure to end with /
files_cnt=2213                  # 2213
sampling_cnt=4000               # 4000
platform="cuda"                 #"cuda" or 'llvm', Used to read json files.
dataset_type="train"            # "train", "test"


if [ "$dataset_type" = "test" ]; then
    files_cnt=95
    sampling_cnt=4000
fi

if [ -d "dataset" ]; then
    rm -rf "dataset"
    echo "Deleted dataset folder."
fi



if [ "$platform" = "cuda" ]; then
    ln -s /mnt/sdb/qinxinghe/dataset_gpu dataset
elif [ "$platform" = "llvm" ]; then
    ln -s /mnt/sdb/qinxinghe/dataset_cpu dataset
else
    echo "Error: Unsupported platform."
fi



python $make_dataset_script \
    --files_cnt $files_cnt \
    --json_files_path $json_files_path \
    --platform $platform \
    --sampling_cnt $sampling_cnt \
    --dataset_type $dataset_type \
    --output_path $output_path

