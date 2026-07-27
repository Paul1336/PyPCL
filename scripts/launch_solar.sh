#!/usr/bin/env bash
# Launch a single tmux session for SoLar comparison (single GPU)
# Usage: bash scripts/launch_solar.sh [GPU_ID]
# Default GPU: 0

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="PyPCL"
SCRIPT="scripts/run_solar_comparison.py"
PHYS_GPU="${1:-0}"
SESSION="solar_gpu${PHYS_GPU}"

echo "Working dir: $WORKDIR"
echo "Conda env:   $CONDA_ENV"
echo ""

tmux kill-session -t "$SESSION" 2>/dev/null
tmux new-session -d -s "$SESSION"

tmux send-keys -t "$SESSION" "cd \"$WORKDIR\"" Enter
tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter
tmux send-keys -t "$SESSION" \
    "CUDA_VISIBLE_DEVICES=${PHYS_GPU} python ${SCRIPT}" \
    Enter

echo "  ${SESSION}  →  GPU ${PHYS_GPU}  (SoLar)"
echo ""
echo "Attach:  tmux attach -t ${SESSION}"
echo "Kill:    tmux kill-session -t ${SESSION}"
