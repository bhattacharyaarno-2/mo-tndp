#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <qlearning|deep_rl> <env> <episodes> <seed_start> <seed_count>}"
ENV_NAME="${2:?Usage: $0 <qlearning|deep_rl> <env> <episodes> <seed_start> <seed_count>}"
EPISODES="${3:?Usage: $0 <qlearning|deep_rl> <env> <episodes> <seed_start> <seed_count>}"
SEED_START="${4:?Usage: $0 <qlearning|deep_rl> <env> <episodes> <seed_start> <seed_count>}"
SEED_COUNT="${5:?Usage: $0 <qlearning|deep_rl> <env> <episodes> <seed_start> <seed_count>}"

if [ "$SEED_COUNT" -lt 1 ]; then
    echo "seed_count must be >= 1"
    exit 1
fi

ARRAY_END=$((SEED_COUNT - 1))
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

case "$MODEL" in
    qlearning)
        DEFAULT_PARTITION=rome
        DEFAULT_TIME=120:00:00
        DEFAULT_GPUS_PER_NODE=
        DEFAULT_CPUS_PER_TASK=16
        ;;
    deep_rl)
        DEFAULT_PARTITION=gpu_mig
        DEFAULT_TIME=120:00:00
        DEFAULT_GPUS_PER_NODE=1
        DEFAULT_CPUS_PER_TASK=8
        ;;
    *)
        echo "Unknown model: $MODEL"
        exit 1
        ;;
esac

PARTITION="${PARTITION:-$DEFAULT_PARTITION}"
TIME="${TIME:-$DEFAULT_TIME}"
GPUS_PER_NODE="${GPUS_PER_NODE:-$DEFAULT_GPUS_PER_NODE}"
CPUS_PER_TASK="${CPUS_PER_TASK:-$DEFAULT_CPUS_PER_TASK}"

SBATCH_ARGS=(
    --partition="$PARTITION"
    --time="$TIME"
    --cpus-per-task="$CPUS_PER_TASK"
    --export=ALL
    --array="0-${ARRAY_END}"
)

if [ -n "$GPUS_PER_NODE" ]; then
    SBATCH_ARGS+=(--gpus-per-node="$GPUS_PER_NODE")
fi

sbatch "${SBATCH_ARGS[@]}" \
    "$SCRIPT_DIR/train_array.sbatch" \
    "$MODEL" "$ENV_NAME" "$EPISODES" "$SEED_START" "$SEED_COUNT"
