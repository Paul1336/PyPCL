#!/usr/bin/env bash
# Launch tmux sessions for GPU 4-7 (SCL-NL, PiCO, PiCO-MCL, ComCo)
# Run from anywhere inside the PyPCL repo:
#   bash scripts/launch_gpu_4to7.sh

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="PyPCL"
SCRIPT="scripts/run_adam_comparison.py"

echo "Working dir: $WORKDIR"
echo "Conda env:   $CONDA_ENV"
echo ""

for GPU_ID in 4 5 6 7; do
    SESSION="adam_gpu${GPU_ID}"

    tmux kill-session -t "$SESSION" 2>/dev/null
    tmux new-session -d -s "$SESSION"

    tmux send-keys -t "$SESSION" "cd \"$WORKDIR\"" Enter
    tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter
    tmux send-keys -t "$SESSION" "CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT} --gpu_id ${GPU_ID}" Enter

    echo "  adam_gpu${GPU_ID}  →  GPU ${GPU_ID}"
done

echo ""
echo "Attach:   tmux attach -t adam_gpu4   (or gpu5 / gpu6 / gpu7)"
echo "List:     tmux ls"
echo "Kill all: tmux kill-session -t adam_gpu4; tmux kill-session -t adam_gpu5; tmux kill-session -t adam_gpu6; tmux kill-session -t adam_gpu7"
