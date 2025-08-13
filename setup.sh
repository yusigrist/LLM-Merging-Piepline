#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

chmod +x ./eval-harness.sh
chmod +x ./holmes-evaluation.sh
# -------------------------------
# Setup lm_eval_harness environment
# -------------------------------
python3 -m venv ./environments/lm_eval_harness_env
source ./environments/lm_eval_harness_env/bin/activate
rm -rf lm-evaluation-harness
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e .
pip install -e ".[math,ifeval,sentencepiece]"
pip install bitsandbytes
cd ..
deactivate  # Optional: exit the lm_eval_harness virtual environment

# -------------------------------
# Setup holmes environment
# -------------------------------
python3 -m venv ./environments/holmes_env
source ./environments/holmes_env/bin/activate
rm -rf holmes-evaluation-BA-FS2025
git clone -b lift-branch https://github.com/yusigrist/holmes-evaluation-BA-FS2025.git
cd ./holmes-evaluation-BA-FS2025
pip install -r requirements.txt
pip install bitsandbytes
cd ./data/flash-holmes
python3 download.py
mv ./flash-holmes-2/* ./
rm -rf flash-holmes-2
rm -rf __MACOSX
cd ../../../
deactivate  # Optional: exit the holmes virtual environment

# -------------------------------
# Setup mergekit environment
# -------------------------------
python3 -m venv ./environments/mergekit_env
source ./environments/mergekit_env/bin/activate
rm -rf mergekit
git clone https://github.com/arcee-ai/mergekit.git
cd mergekit
pip install -e .
pip install bitsandbytes
cd ..
mergekit-yaml --help
