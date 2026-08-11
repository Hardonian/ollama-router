#!/usr/bin/env bash
# pull-model.sh — orchestrate model pulls across GPU lanes
# Usage: pull-model.sh <model_name> [lane_name]

set -euo pipefail

MODEL="${1:-}"
LANE="${2:-}"

if [[ -z "$MODEL" ]]; then
    echo "Usage: $0 <model_name> [lane_name]"
    exit 1
fi

LANES=(
    "ollama-v100:11437:0"
    "ollama-p40:11435:1"
    "ollama-3060:11436:2"
    "ollama-default:11434:0"
)

if [[ -n "$LANE" ]]; then
    LANES=($(printf '%s\n' "${LANES[@]}" | grep "^${LANE}:"))
fi

for entry in "${LANES[@]}"; do
    IFS=':' read -r unit port gpu <<< "$entry"
    echo "Pulling $MODEL to $unit (GPU $gpu, port $port)..."
    CUDA_VISIBLE_DEVICES="$gpu" OLLAMA_VULKAN=false ollama pull "$MODEL" &
done

wait
echo "All pulls completed."