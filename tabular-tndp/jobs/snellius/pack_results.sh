#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."

STAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE="${1:-snellius-results-${STAMP}.tar.gz}"

INCLUDE=()
for path in q_tables deep_models wandb eval carbon_logs logs; do
    if [ -e "$path" ]; then
        INCLUDE+=("$path")
    fi
done

if [ "${#INCLUDE[@]}" -eq 0 ]; then
    echo "No result directories found."
    exit 1
fi

tar -czf "$ARCHIVE" "${INCLUDE[@]}"
tar -tzf "$ARCHIVE" > "${ARCHIVE%.tar.gz}.manifest.txt"

echo "Created $ARCHIVE"
echo "Created ${ARCHIVE%.tar.gz}.manifest.txt"
ls -lh "$ARCHIVE" "${ARCHIVE%.tar.gz}.manifest.txt"
