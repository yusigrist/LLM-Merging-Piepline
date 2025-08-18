#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

export RAY_USE_MULTIPROCESSING_CPU_COUNT=
export CUDA_LAUNCH_BLOCKING=0
export TORCH_USE_CUDA_DSA=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export RAY_TMPDIR=/local/ray

################################################################################
# Step 1: Install required Python packages
################################################################################

echo "Installing required packages..."
pip install transformers torch torchvision torchaudio accelerate ipywidgets bitsandbytes

################################################################################
# Step 2: Define variables (equivalent to the Python script)
################################################################################

# Models to evaluate from Hugging Face
models=(
  "Qwen/Qwen2.5-Coder-7B"
)

do_holmes=false
harness_batch_size=16
harness_batch_size_moe=6
holmes_batch_size=8
holmes_batch_size_moe=8

# Tasks for LM Evaluation Harness for each model
do_harness=("gsm8k" "leaderboard" "mmlu")

# Hugging Face credentials
huggingface_api_key="api_key_placeholder"  # Replace with your actual Hugging Face API key
huggingface_username="your_username"  # Replace with your actual Hugging Face username

################################################################################
# Step 3: Hugging Face login
################################################################################

echo "Logging into Hugging Face..."
huggingface-cli login --token "${huggingface_api_key}"

################################################################################
# Helper function to run commands with real-time output
################################################################################

run_command() {
    local cmd="$1"
    echo "------------------------------------------------------------------------------"
    echo "Running command: $cmd"
    echo "------------------------------------------------------------------------------"
    eval "$cmd"
    echo "------------------------------------------------------------------------------"
}

################################################################################
# Step 4: Run LM Evaluation Harness for models in parallel
################################################################################

echo "Starting LM Evaluation Harness evaluations for models..."
harness_pids=()

for i in "${!models[@]}"; do
    model="${models[$i]}"
    # choose batch size based on whether model name contains "moe"
    if [[ "$model" == *moe* ]]; then
      bs=$harness_batch_size_moe
    else
      bs=$harness_batch_size
    fi

    (
      for harness_task in "${do_harness[@]}"; do
          run_command "./eval-harness.sh ${model} ${harness_task} false $((i)) ${bs}"
      done
    ) &
    harness_pids+=($!)
done

echo "Waiting for LM Evaluation Harness evaluations for models..."
for pid in "${harness_pids[@]}"; do
    wait "$pid"
done
echo "All LM Evaluation Harness evaluations for models completed!"

################################################################################
# Step 5: Run Holmes evaluations for models in parallel
################################################################################

if do_holmes; then
    echo "Starting Holmes evaluations for models..."
    holmes_pids=()
    
    for i in "${!models[@]}"; do
        model="${models[$i]}"
        # choose Holmes batch size based on whether model name contains "moe"
        if [[ "$model" == *moe* ]]; then
          hbs=$holmes_batch_size_moe
          ps="four_bit"
        else
          hbs=$holmes_batch_size
          ps="full"
        fi
    
        (
          run_command "./holmes-evaluation.sh ${model} flash-holmes $((i)) ${hbs} ${ps}"
        ) &
        holmes_pids+=($!)
    done
    
    echo "Waiting for Holmes evaluations for models..."
    for pid in "${holmes_pids[@]}"; do
        wait "$pid"
    done
    echo "All Holmes evaluations for models completed!"
fi
echo "All done!"
