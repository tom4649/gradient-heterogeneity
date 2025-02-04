#!/bin/bash
dataset_name=$1
optimizer_name=$2
model_name=$3
domain=$4
training_mode=$5

if [ -z "$training_mode" ]; then
  training_mode=normal
fi

echo "Start hessian_per_param: $dataset_name $optimizer_name $model_name $domain $training_mode"

cmd="python3 hessian_per_param.py \
--optimizer_name $optimizer_name \
--dataset_name $dataset_name \
--model_name $model_name \
--domain $domain \
--training_mode $training_mode"
echo $cmd
eval $cmd
