#!/bin/bash
script_type=evaluation
hyperparameter_config_dir=conf

dataset_name=$1
optimizer_name=$2
model_name=$3
lr_scheduler_type=$4

if [ -z "$lr_scheduler_type" ]; then
  lr_scheduler_type=default
fi

echo "Start training: $dataset_name $optimizer_name $model_name $lr_scheduler_type"

for seed in 1111 2222 3333 4444 5555; do
  cmd="python3 train_nlp.py \
    --optimizer_name $optimizer_name \
    --dataset_name $dataset_name \
    --seed $seed \
    --script_type $script_type \
    --hyperparameter_config_dir $hyperparameter_config_dir \
    --model_name $model_name \
    --lr_scheduler_type $lr_scheduler_type"
  echo $cmd
  eval $cmd
done
