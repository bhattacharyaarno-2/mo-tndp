#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${MO_TNDP_ENV:-mo-tndp}"

cd "$(dirname "$0")/../../.."

if command -v module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1 || true
    if [ -n "${CONDA_MODULE:-}" ]; then
        module load "$CONDA_MODULE"
    else
        for stack in 2025 2024 2023 2022; do
            module load "$stack" >/dev/null 2>&1 || true
            for candidate in Miniconda3 Anaconda3 Mamba miniconda3 anaconda3 mamba; do
                module load "$candidate" >/dev/null 2>&1 && break 2
            done
            module purge >/dev/null 2>&1 || true
        done
    fi
fi

if command -v mamba >/dev/null 2>&1; then
    CONDA_BIN=mamba
elif command -v conda >/dev/null 2>&1; then
    CONDA_BIN=conda
else
    echo "Could not find conda or mamba."
    echo "Run one of these on Snellius to find the exact module name:"
    echo "  module spider Miniconda3"
    echo "  module spider Anaconda3"
    echo "  module spider Mamba"
    echo "Then retry, for example:"
    echo "  CONDA_MODULE='Miniconda3/<version-or-full-path-from-module-spider>' bash tabular-tndp/jobs/snellius/setup_env.sh"
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
