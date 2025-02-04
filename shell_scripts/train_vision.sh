#!/bin/bash
script_type=evaluation
hyperparameter_config_dir=conf

dataset_name=$1
optimizer_name=$2
model_name=$3

echo "Start training : $model_name $dataset_name $optimizer_name"

for seed in 1111 2222 3333 4444 5555; do
    cmd="python3 train_vision.py \
    --optimizer_name $optimizer_name \
    --dataset_name $dataset_name \
    --model_name $model_name \
    --script_type $script_type \
    --domain vision \
    --hyperparameter_config_dir $hyperparameter_config_dir \
    --seed $seed"

    echo $cmd

    eval $cmd
done
