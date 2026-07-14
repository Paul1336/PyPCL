#!/usr/bin/env bash
# Launch tmux sessions for GPU 0-3 (Cour2011, Wu2022, PRODEN, MCL-LOG)
# Run from anywhere inside the PyPCL repo:
#   bash scripts/launch_gpu_0to3.sh

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="PyPCL"
SCRIPT="scripts/run_adam_comparison.py"

echo "Working dir: $WORKDIR"
echo "Conda env:   $CONDA_ENV"
echo ""

for GPU_ID in 0 1 2 3; do
    SESSION="adam_gpu${GPU_ID}"

    tmux kill-session -t "$SESSION" 2>/dev/null
    tmux new-session -d -s "$SESSION"

    tmux send-keys -t "$SESSION" "cd \"$WORKDIR\"" Enter
    tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter
    tmux send-keys -t "$SESSION" "CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT} --gpu_id ${GPU_ID}" Enter

    echo "  adam_gpu${GPU_ID}  →  GPU ${GPU_ID}"
done

echo ""
echo "Attach:   tmux attach -t adam_gpu0   (or gpu1 / gpu2 / gpu3)"
echo "List:     tmux ls"
echo "Kill all: tmux kill-session -t adam_gpu0; tmux kill-session -t adam_gpu1; tmux kill-session -t adam_gpu2; tmux kill-session -t adam_gpu3"
