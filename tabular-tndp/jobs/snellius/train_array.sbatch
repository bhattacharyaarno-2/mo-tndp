#!/usr/bin/env bash
#SBATCH --job-name=mo-tndp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=28G
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --error=logs/slurm-%A_%a.err

set -euo pipefail

MODEL="${1:?MODEL missing: qlearning or deep_rl}"
ENV_NAME="${2:?ENV missing, e.g. amsterdam}"
TRAIN_EPISODES="${3:?TRAIN_EPISODES missing}"
SEED_START="${4:?SEED_START missing}"
SEED_COUNT="${5:?SEED_COUNT missing}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED=$((SEED_START + TASK_ID))
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
CONDA_ENV="${MO_TNDP_ENV:-mo-tndp}"
WANDB_MODE="${WANDB_MODE:-online}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs q_tables deep_models eval carbon_logs wandb

if command -v module >/dev/null 2>&1; then
    module purge >/dev/null 2>&1 || true
    if [ -n "${CONDA_MODULE:-}" ]; then
        if [ -n "${CONDA_MODULE_STACK:-}" ]; then
            module load "$CONDA_MODULE_STACK"
        fi
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
    echo "Could not find conda or mamba. Set CONDA_MODULE to the exact module from 'module spider Miniconda3'."
    exit 1
fi

eval "$("$CONDA_BIN" shell.bash hook)"
conda activate "$CONDA_ENV"

export WANDB_MODE
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "host=$(hostname)"
echo "model=$MODEL env=$ENV_NAME episodes=$TRAIN_EPISODES seed=$SEED task=$TASK_ID/$SEED_COUNT"
echo "partition=${SLURM_JOB_PARTITION:-unknown} cpus=${SLURM_CPUS_PER_TASK:-unknown} cuda=${CUDA_VISIBLE_DEVICES:-none} wandb=$WANDB_MODE"
python - <<'PY'
import torch
print("torch_cuda_available=", torch.cuda.is_available(), "cuda_device_count=", torch.cuda.device_count(), flush=True)
PY

case "$ENV_NAME" in
    amsterdam)
        NR_STATIONS="${NR_STATIONS:-30}"
        NR_GROUPS="${NR_GROUPS:-5}"
        ;;
    amsterdam_10x10)
        NR_STATIONS="${NR_STATIONS:-30}"
        NR_GROUPS="${NR_GROUPS:-5}"
        ;;
    xian)
        NR_STATIONS="${NR_STATIONS:-47}"
        NR_GROUPS="${NR_GROUPS:-5}"
        ;;
    *)
        NR_STATIONS="${NR_STATIONS:-9}"
        NR_GROUPS="${NR_GROUPS:-5}"
        ;;
esac

REWARD_TYPE="${REWARD_TYPE:-ggi2}"
OD_TYPE="${OD_TYPE:-pct}"
TEST_EPISODES="${TEST_EPISODES:-3}"
EPSILON_DECAY_STEPS="${EPSILON_DECAY_STEPS:-$TRAIN_EPISODES}"
EPSILON_WARMUP_STEPS="${EPSILON_WARMUP_STEPS:-0}"

if [ "$MODEL" = "qlearning" ]; then
    RUN_ID="${RUN_ID_PREFIX:-snellius-q}-${ENV_NAME}-${REWARD_TYPE}-${SEED}-${RUN_STAMP}"
    CMD=(
        python tabular-tndp/train_qlearning.py
        --env "$ENV_NAME" \
        --nr_stations "$NR_STATIONS" \
        --nr_groups "$NR_GROUPS" \
        --train_episodes "$TRAIN_EPISODES" \
        --epsilon_decay_steps "$EPSILON_DECAY_STEPS" \
        --epsilon_warmup_steps "$EPSILON_WARMUP_STEPS" \
        --test_episodes "$TEST_EPISODES" \
        --initial_epsilon "${INITIAL_EPSILON:-1}" \
        --final_epsilon "${FINAL_EPSILON:-0.02}" \
        --alpha "${ALPHA:-0.1}" \
        --gamma "${GAMMA:-0.99}" \
        --od_type "$OD_TYPE" \
        --chained_reward \
        --reward_type "$REWARD_TYPE" \
        --exploration_type "${EXPLORATION_TYPE:-egreedy}" \
        --update_method "${UPDATE_METHOD:-mc}" \
        --seed "$SEED" \
        --run_id "$RUN_ID"
    )
    if [ -n "${NO_LOG:-}" ]; then
        CMD+=(--no_log)
    fi
    if [ -n "${IGNORE_EXISTING_LINES:-}" ]; then
        CMD+=(--ignore_existing_lines)
    fi
    "${CMD[@]}"
elif [ "$MODEL" = "deep_rl" ]; then
    RUN_NAME="${RUN_NAME_PREFIX:-snellius-dqn}-${ENV_NAME}-${REWARD_TYPE}-${SEED}-${RUN_STAMP}"
    CMD=(
        python tabular-tndp/train_deep_rl.py
        --env "$ENV_NAME" \
        --nr_stations "$NR_STATIONS" \
        --nr_groups "$NR_GROUPS" \
        --train_episodes "$TRAIN_EPISODES" \
        --epsilon_decay_steps "$EPSILON_DECAY_STEPS" \
        --epsilon_warmup_steps "$EPSILON_WARMUP_STEPS" \
        --test_episodes "$TEST_EPISODES" \
        --initial_epsilon "${INITIAL_EPSILON:-1}" \
        --final_epsilon "${FINAL_EPSILON:-0.02}" \
        --gamma "${GAMMA:-0.99}" \
        --od_type "$OD_TYPE" \
        --chained_reward \
        --reward_type "$REWARD_TYPE" \
        --seed "$SEED" \
        --run_name "$RUN_NAME" \
        --no_data_penalty "${NO_DATA_PENALTY:-0}" \
        --existing_connection_bonus "${EXISTING_CONNECTION_BONUS:-0}" \
        --existing_overlap_penalty "${EXISTING_OVERLAP_PENALTY:-0}"
    )
    if [ -n "${DEVICE:-}" ]; then
        CMD+=(--device "$DEVICE")
    fi
    if [ -n "${NO_LOG:-}" ]; then
        CMD+=(--no_log)
    fi
    if [ -n "${IGNORE_EXISTING_LINES:-}" ]; then
        CMD+=(--ignore_existing_lines)
    fi
    "${CMD[@]}"
else
    echo "Unknown MODEL: $MODEL"
    exit 1
fi
