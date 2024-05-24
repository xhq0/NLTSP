#!/bin/bash

job="eval"             # "train", "eval"
model_name="lstm"

num_workers=0

case ${model_name} in
    "lstm")
        num_axes=12
        num_epochs=50
        batch_size=2048
        learning_rate=0.002
        gamma=0.8 # 0.9
        weight_decay=1e-6
        scheduler_step_size=10 # 15
        ;;
    *)
        num_epochs=500
        learning_rate=0.00001
        ;;
esac



if [[ $job == "train" ]]; then
    device="0"
    dataset="pkl_dataset/nltsp_dataset_t4_2213_4000_train.pkl"
    output_path="trained_model/nltsp-t4-amp"
    train_script="nltsp_train_single_gpu.py"
    
    CUDA_VISIBLE_DEVICES=$device \
    python $train_script \
        --dataset $dataset \
        --lr $learning_rate \
        --epochs $num_epochs \
        --batch_size $batch_size \
        --weight_decay $weight_decay \
        --num_workers $num_workers \
        --model_name $model_name \
        --output_path $output_path \
        --gamma $gamma \
        --scheduler_step_size $scheduler_step_size \
        --num_axes $num_axes


elif [[ $job == "eval" ]]; then
    device=0
    dataset="pkl_dataset/nltsp_dataset_platinum-8272_95_4000_test.pkl"
    trained_model="trained_model/nltsp-t4/nltsp_model_40.pkl"
    platform="llvm"
    eval_script="nltsp_eval.py"

    python $eval_script \
        --dataset $dataset \
        --trained_model $trained_model \
        --platform $platform \
        --device $device \
        --lr $learning_rate \
        --epochs $num_epochs \
        --batch_size $batch_size \
        --weight_decay $weight_decay \
        --num_workers $num_workers \
        --model_name $model_name \
        --num_axes $num_axes

else
    echo "Error: Unknown job type. Please provide 'train' or 'eval' as job type."
    exit 1
fi


