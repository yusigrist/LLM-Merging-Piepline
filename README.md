# LLM Merging & Evaluation Pipeline

This pipeline provides a comprehensive toolkit to merge, evaluate, and analyze Large Language Models (LLMs). It leverages powerful open-source tools like `mergekit` for model merging and `lm-evaluation-harness` & `holmes-evaluation` for robust performance assessment. The entire process, from merging to results, is automated through a series of configurable shell scripts.

## 🚀 Features

* **Model Merging**: Supports various merging techniques including `linear`, `slerp`, `ties`, `dare_ties`, `task_arithmetic` etc.
* **Comprehensive Evaluation**: Utilizes standard academic benchmarks through `lm-evaluation-harness` (e.g., MMLU, GSM8K, Leaderboard) and linguistic capability probing with `holmes-evaluation` (Flash-Holmes).
* **Automated Workflow**: Streamlines the end-to-end process of merging, uploading to Hugging Face, and evaluating with simple script execution.
* **Parallel Processing**: Efficiently runs evaluations on multiple models in parallel across different GPUs.

---

## 📂 Project Structure

```

.
├── environments/         \# Stores Python virtual environments
├── merge\_config/         \# Default directory for mergekit configuration files
├── harness\_results/      \# Raw output from lm-evaluation-harness
├── holmes\_results/       \# Raw output from holmes-evaluation
├── organized\_results/    \# Cleaned and organized evaluation results
├── setup.sh              \# 🚀 **START HERE**: Installs all dependencies and sets up environments
├── script.sh             \# Main script for a full merge, upload, and evaluate pipeline
├── merge\_only.sh         \# Script to only merge models and upload them
├── evaluation\_only.sh    \# Script to only evaluate existing models
└── eval-harness.sh       \# Helper script to run lm-evaluation-harness

````

---

## 🛠️ Setup

Before running any scripts, you need to set up the necessary environments and dependencies. The `setup.sh` script automates this entire process.

1.  **Make the script executable**:
    ```bash
    chmod +x setup.sh
    ```

2.  **Run the setup script**:
    ```bash
    ./setup.sh
    ```

This script will:
* Create three separate Python virtual environments in the `./environments/` directory for `mergekit`, `lm-evaluation-harness`, and `holmes-evaluation`.
* Clone the required repositories.
* Install all necessary Python packages within their respective environments.

---

## ⚙️ How to Use

The pipeline is operated through three main shell scripts: `script.sh`, `merge_only.sh`, and `evaluation_only.sh`. Before running, you **must** configure the variables within the script you choose to use.

### 1. Full Pipeline: Merge & Evaluate (`script.sh`)

This is the main script to perform the entire workflow: merge models, upload the result to the Hugging Face Hub, and then run evaluations.

**Configuration:**
Open `script.sh` and modify the variables in `Step 2`, including:
* `model_1`, `model_2`, `base_model`: The Hugging Face model identifiers you want to merge.
* `merge_methods`: An array of methods to use for merging (e.g., `"linear" "slerp"`).
* `huggingface_api_key` and `huggingface_username`: Your Hugging Face credentials.
* `do_merge_model_holmes` and `do_merge_model_harness`: Set which evaluations to run on the newly merged models.

**Execution:**
```bash
./script.sh
````

### 2\. Merge Only (`merge_only.sh`)

Use this script if you only want to merge models and upload them to the Hugging Face Hub without running evaluations.

**Configuration:**
Open `merge_only.sh` and set the `model_1`, `model_2`, `base_model`, `merge_methods`, and your Hugging Face credentials.

**Execution:**

```bash
./merge_only.sh
```

### 3\. Evaluate Only (`evaluation_only.sh`)

Use this script to evaluate models that are already on the Hugging Face Hub (or available locally).

**Configuration:**
Open `evaluation_only.sh` and set:

  * `models`: An array of Hugging Face model identifiers to evaluate.
  * `do_harness`: An array of `lm-evaluation-harness` tasks to run.
  * `do_holmes`: Set to `true` or `false` to run Holmes evaluation.
  * Your Hugging Face credentials.

**Execution:**

```bash
./evaluation_only.sh
```

-----

