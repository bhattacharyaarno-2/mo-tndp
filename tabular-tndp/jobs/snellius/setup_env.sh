#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${MO_TNDP_ENV:-mo-tndp}"

cd "$(dirname "$0")/../../.."

if command -v module >/dev/null 2>&1; then
    module purge || true
    module load 2023 || true
    module load Miniconda3 || module load Anaconda3 || module load Mamba || module load miniforge || true
fi

if command -v mamba >/dev/null 2>&1; then
    CONDA_BIN=mamba
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN=conda
else
    echo "Could not find conda or mamba. Run: module avail 2>&1 | grep -Ei 'mamba|conda|miniforge|python'"
    exit 1
fi

eval "$("$CONDA_BIN" shell.bash hook)"

if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    "$CONDA_BIN" env create -n "$ENV_NAME" -f tabular-tndp/environment.yml
fi

conda activate "$ENV_NAME"
python -m pip install -e .
python - <<'PY'
import mo_gymnasium
import numpy
import torch
import wandb
import motndp

print("Environment OK")
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY
