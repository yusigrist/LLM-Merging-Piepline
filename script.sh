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
# Step 2: Define variables 
################################################################################

# Define models and tasks for evaluations
model_1="Qwen/Qwen2.5-Coder-7B"
do_model_1_holmes=false
# Tasks for LM Evaluation Harness for model_1:
do_model_1_harness=()

model_2="Qwen/Qwen2.5-Math-7B"
do_model_2_holmes=false
do_model_2_harness=()

base_model="Qwen/Qwen2.5-7B"
do_base_holmes=false
do_base_harness=()

# Set to true to run Holmes evaluations in parallel for base models.
parallel_holmes=true

harness_batch_size=16
harness_batch_size_moe=6

holmes_batch_size=10
holmes_batch_size_moe=5

# Merge configuration
mergekit_config_path="./merge_config/config.yml"
model_output_path="./output_model"
# List of merge methods to iterate over
merge_methods=("linear" "slerp" "moe" "ties")
do_moe=false
gate_mode="random"

# Hugging Face credentials
huggingface_api_key="api_key_placeholder"  # Replace with your actual Hugging Face API key
huggingface_username="your_username"  # Replace with your actual Hugging Face username

# For merged models evaluations (Holmes and harness)
do_merge_model_holmes=true
do_merge_model_harness=("leaderboard" "mmlu" "gsm8k")

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
# Step 4: Run Holmes evaluations for base models in parallel (using GPUs 0-2)
################################################################################

if [ "$parallel_holmes" = false ]; then
    echo "Running Holmes evaluations sequentially to avoid OOM..."
    if [ "$do_model_1_holmes" = true ]; then
        run_command "./holmes-evaluation.sh ${model_1} flash-holmes 0"
    fi
    if [ "$do_model_2_holmes" = true ]; then
        run_command "./holmes-evaluation.sh ${model_2} flash-holmes 0"
    fi
    if [ "$do_base_holmes" = true ]; then
        run_command "./holmes-evaluation.sh ${base_model} flash-holmes 0"
    fi
else
    echo "Running Holmes evaluations in parallel for base models..."
    pids=()
    if [ "$do_model_1_holmes" = true ]; then
        ( run_command "./holmes-evaluation.sh ${model_1} flash-holmes 0" ) &
        pids+=($!)
    fi
    if [ "$do_model_2_holmes" = true ]; then
        ( run_command "./holmes-evaluation.sh ${model_2} flash-holmes 1" ) &
        pids+=($!)
    fi
    if [ "$do_base_holmes" = true ]; then
        ( run_command "./holmes-evaluation.sh ${base_model} flash-holmes 2" ) &
        pids+=($!)
    fi
    echo "Waiting for parallel Holmes evaluations to finish for base models..."
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    echo "All parallel Holmes evaluations for base models completed!"
fi

################################################################################
# Step 5: Run LM Evaluation Harness for base models sequentially per model
################################################################################

echo "Starting LM Evaluation Harness evaluations for base models..."
base_harness_pids=()

if [ "${#do_model_1_harness[@]}" -ne 0 ]; then
    (
      for harness_task in "${do_model_1_harness[@]}"; do
          run_command "./eval-harness.sh ${model_1} ${harness_task} false 0 ${harness_batch_size}"
      done
    ) &
    base_harness_pids+=($!)
fi

if [ "${#do_model_2_harness[@]}" -ne 0 ]; then
    (
      for harness_task in "${do_model_2_harness[@]}"; do
          run_command "./eval-harness.sh ${model_2} ${harness_task} false 1 ${harness_batch_size}"
      done
    ) &
    base_harness_pids+=($!)
fi

if [ "${#do_base_harness[@]}" -ne 0 ]; then
    (
      for harness_task in "${do_base_harness[@]}"; do
          run_command "./eval-harness.sh ${base_model} ${harness_task} false 2 ${harness_batch_size}"
      done
    ) &
    base_harness_pids+=($!)
fi

echo "Waiting for LM Evaluation Harness evaluations for base models..."
for pid in "${base_harness_pids[@]}"; do
    wait "$pid"
done
echo "All LM Evaluation Harness evaluations for base models completed!"

################################################################################
# Step 6: Loop over merge methods to merge models and run evaluations on merged models
################################################################################

merged_holmes_pids=()
merged_harness_pids=()

# Loop over merge methods (will assign GPU indices 0-3 to each merged model)
for i in "${!merge_methods[@]}"; do
    merge_method="${merge_methods[$i]}"
    echo "Processing merge method: $merge_method"

    # Construct merged model name (e.g., Qwen2.5-Coder-7B-Qwen2.5-Math-7B-Merged-linear-19)
    merged_model_name="$(basename "$model_1")-$(basename "$model_2")-Merged-${merge_method}"

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
models:
  - model: $model_1
  - model: $model_2
merge_method: slerp
base_model: $model_1
parameters:
  t: 0.5
dtype: bfloat16
EOF
    elif [ "$merge_method" = "moe" ]; then
        cat <<EOF > "merge_config_moe/config.yml"
base_model: $base_model
gate_mode: $gate_mode
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
  - model: $model_1
    parameters:
      density: [1, 0.7, 0.1] # density gradient
      weight: 1.0
  - model: $model_2
    parameters:
      density: 0.5
      weight: [0, 0.3, 0.7, 1] # weight gradient
merge_method: ties
base_model: $base_model
parameters:
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




# Start evaluations after all merging processes are done
for i in "${!merge_methods[@]}"; do
    merge_method="${merge_methods[$i]}"
    merged_model_name="$(basename "$model_1")-$(basename "$model_2")-Merged-${merge_method}"

    harness_batch_size_temp=$harness_batch_size
    if [ "$merge_method" = "moe" ]; then
        harness_batch_size_temp=$harness_batch_size_moe
    fi

    # Run LM Evaluation Harness tasks sequentially for the merged model on GPU index $i
    if [ "${#do_merge_model_harness[@]}" -ne 0 ]; then
        (
            for harness_task in "${do_merge_model_harness[@]}"; do
                run_command "./eval-harness.sh ${huggingface_username}/${merged_model_name} ${harness_task} false $((i)) $harness_batch_size_temp"
            done
        ) &
        merged_harness_pids+=($!)
    fi
done
echo "Waiting for LM Evaluation Harness evaluations for merged models to finish..."
for pid in "${merged_harness_pids[@]}"; do
    wait "$pid"
done

# Start evaluations after all merging processes are done
for i in "${!merge_methods[@]}"; do
    merge_method="${merge_methods[$i]}"
    merged_model_name="$(basename "$model_1")-$(basename "$model_2")-Merged-${merge_method}-24"

    holmes_batch_size_temp=$holmes_batch_size
    if [ "$merge_method" = "moe" ]; then
        holmes_batch_size_temp=$holmes_batch_size_moe
    fi

    # Evaluate the merged model with Holmes on GPU index $i (using GPUs 0-3)
    if [ "$do_merge_model_holmes" = true ]; then
        ( run_command "./holmes-evaluation.sh ${huggingface_username}/${merged_model_name} flash-holmes $((i)) ${holmes_batch_size_temp}" ) &
        merged_holmes_pids+=($!)
    fi
    
done

echo "Waiting for parallel Holmes evaluations for merged models to finish..."
for pid in "${merged_holmes_pids[@]}"; do
    wait "$pid"
done
echo "All parallel Holmes evaluations for merged models completed!"

echo "All done!"
