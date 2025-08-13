#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

export RAY_USE_MULTIPROCESSING_CPU_COUNT=
export CUDA_LAUNCH_BLOCKING=0
export CUDA_VISIBLE_DEVICES=1
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

# Models to merge
model_1="Qwen/Qwen2.5-7B-Instruct"
model_2="Qwen/Qwen2.5-Coder-7B"
base_model="Qwen/Qwen2.5-7B-Instruct"

# Merge configuration
mergekit_config_path="./merge_config/config.yml"
model_output_path="./output_model"
# List of merge methods to iterate over
merge_methods=("linear" "slerp" "ties" "dare_ties" "task_arithmetic" "della")
do_moe=false
gate_mode="hidden"

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
# Step 4: Loop over merge methods to merge models
################################################################################

# Loop over merge methods (will assign GPU indices 0-3 to each merged model)
for i in "${!merge_methods[@]}"; do
    merge_method="${merge_methods[$i]}"
    echo "Processing merge method: $merge_method"

    # Construct merged model name (e.g., Qwen2.5-Coder-7B-Qwen2.5-Math-7B-Merged-linear-19)
    merged_model_name="$(basename "$model_1")-$(basename "$model_2")-Merged-${merge_method}-29"

    # Create the MergeKit config file based on merge_method
    mkdir -p "$(dirname "$mergekit_config_path")"
    mkdir -p "merge_config_${merge_method}"
    echo "Preparing MergeKit config for merge method: $merge_method..."

    if [ "$merge_method" = "linear" ]; then
        cat <<EOF > "merge_config_linear/config.yml"
models:
- model: $model_1
  parameters:
    weight: 0.5
- model: $model_2
  parameters:
    weight: 0.5
merge_method: linear
dtype: bfloat16
EOF
    elif [ "$merge_method" = "slerp" ]; then
        cat <<EOF > "merge_config_slerp/config.yml"
base_model: $base_model
dtype: bfloat16
merge_method: slerp
parameters:
  t: 0.1
slices:
- sources:
  - layer_range: [0, 28]
    model: $model_1
  - layer_range: [0, 28]
    model: $model_2
EOF
    elif [ "$merge_method" = "moe" ]; then
        cat <<EOF > "merge_config_moe/config.yml"
base_model: $base_model
dtype: bfloat16
experts:
  - source_model: $model_1
    positive_prompts:
      - "code"
      - "python"
      - "javascript"
      - "programming"
      - "algorithm"
  - source_model: $model_2
    positive_prompts:
      - "math"
      - "calculus"
      - "algebra"
      - "geometry"
      - "trigonometry"
      - "reason"
      - "mathematics"
      - "solve"
      - "count"
shared_experts:
  - source_model: $base_model
    positive_prompts:
      - "chat"
      - "assistant"
      - "fact"
    residual_scale: 0.1
EOF
    elif [ "$merge_method" = "ties" ]; then
        cat <<EOF > "merge_config_ties/config.yml"
models:
  - model: $base_model
  - model: $model_1
    parameters:
     weight: 0.5
     density: 0.5
  - model: $model_2
    parameters:
     weight: 0.5
     density: 0.5
merge_method: ties
base_model: $base_model
parameters:
 normalize: false
 int8_mask: true
dtype: float16
EOF
    elif [ "$merge_method" = "dare_ties" ]; then
        cat <<EOF > "merge_config_dare_ties/config.yml"
models:
  - model: $model_1
    parameters:
      weight: 0.5
      density: 0.8
  - model: $model_2
    parameters:
      weight: 0.5
      density: 0.8
merge_method: dare_ties
base_model: $base_model
dtype: bfloat16

EOF
    elif [ "$merge_method" = "task_arithmetic" ]; then
        cat <<EOF > "merge_config_task_arithmetic/config.yml"
base_model: $base_model
dtype: bfloat16
merge_method: task_arithmetic
parameters:
  lambda: 0.5676097213578511
  normalize: 1.0
slices:
- sources:
  - layer_range: [0, 28]
    model: $base_model
  - layer_range: [0, 28]
    model: $model_1
    parameters:
      weight: 0.5
  - layer_range: [0, 28]
    model: $model_2
    parameters:
      weight: 0.5
EOF
    elif [ "$merge_method" = "della" ]; then
        cat <<EOF > "merge_config_della/config.yml"
models:
  - model: $model_1
    parameters:
      weight: 0.5
  - model: $model_2
    parameters:
      weight: 0.6
merge_method: della
base_model: $base_model
parameters:
  density: 0.8
  normalize: true
  int8_mask: true
dtype: float16
EOF
    fi

    echo "MergeKit config saved to: $mergekit_config_path"
    echo "Full config contents:"
    cat "$mergekit_config_path"

    mkdir -p "${model_output_path}_${merge_method}"

    # Run MergeKit to merge models sequentially (not in parallel)
    if [ "$merge_method" = "moe" ]; then
        echo "Merging models via MergeKit MoE for merge method: $merge_method..."
        run_command "source ./environments/mergekit_env/bin/activate && mergekit-moe merge_config_${merge_method}/config.yml ${model_output_path}_${merge_method}"
    else
        echo "Merging models via MergeKit for merge method: $merge_method..."
        run_command "source ./environments/mergekit_env/bin/activate && mergekit-yaml merge_config_${merge_method}/config.yml ${model_output_path}_${merge_method} --cuda"
    fi

    # Create and upload the merged model repository on Hugging Face
    echo "Creating Hugging Face repo: ${merged_model_name}"
    huggingface-cli repo create "${merged_model_name}" -y
    echo "Uploading merged model to Hugging Face for merge method: $merge_method..."
    run_command "huggingface-cli upload '${huggingface_username}/${merged_model_name}' '${model_output_path}_${merge_method}'"
    echo "Upload finished for ${merge_method}. Cleaning up output directory: ${model_output_path}"
    # WARNING: rm -rf permanently deletes files/directories. Ensure the path is correct.
    rm -rf "${model_output_path}_${merge_method}"
    echo "Output directory cleaned."
done

echo "All merging processes completed!"