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
        ;;
    deep_rl)
        DEFAULT_PARTITION=gpu_mig
        DEFAULT_TIME=120:00:00
        ;;
    *)
        echo "Unknown model: $MODEL"
        exit 1
        ;;
esac

PARTITION="${PARTITION:-$DEFAULT_PARTITION}"
TIME="${TIME:-$DEFAULT_TIME}"

sbatch \
    --partition="$PARTITION" \
    --time="$TIME" \
    --array="0-${ARRAY_END}" \
    "$SCRIPT_DIR/train_array.sbatch" \
    "$MODEL" "$ENV_NAME" "$EPISODES" "$SEED_START" "$SEED_COUNT"
