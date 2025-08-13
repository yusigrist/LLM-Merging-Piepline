#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# get model_name, version from arguments
model_name=$1
dataset=$2
use_chat_template=$3
gpu=$4
batch_size=$5


source ./environments/lm_eval_harness_env/bin/activate

# if use_chat_template is true, use add --apply_chat_template
if [ "$use_chat_template" = "true" ]; then
    python3 -m lm_eval --model hf \
        --model_args pretrained=$model_name,load_in_4bit=True\
        --tasks $dataset \
        --device cuda:$gpu \
        --apply_chat_template \
        --batch_size auto \
        --log_samples \
        --output_path ./harness_results/$model_name/$dataset
    exit 0
fi
lm_eval --model hf \
    --model_args pretrained=$model_name,load_in_4bit=True\
    --tasks $dataset \
    --batch_size auto \
    --device cuda:$gpu \
    --log_samples \
    --output_path ./harness_results/$model_name/$dataset
exit 0
