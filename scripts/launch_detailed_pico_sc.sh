#!/usr/bin/env bash
# Rerun PiCO-SC detailed analysis on GPU 2 (k=5,10,15 first, then rest of 5-15).
# Usage: bash scripts/launch_detailed_pico_sc.sh

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="PyPCL"
SCRIPT="scripts/run_detailed_analysis.py"
GPU_ID=2
C=20
SESSION="detail_pico_sc"

echo "Working dir: $WORKDIR"
echo "Session:     $SESSION  →  GPU ${GPU_ID}  alg=PiCO-SC"

tmux kill-session -t "$SESSION" 2>/dev/null
tmux new-session -d -s "$SESSION"

tmux send-keys -t "$SESSION" "cd \"$WORKDIR\"" Enter
tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter

for K in 5 10 15 6 7 8 9 11 12 13 14; do
    tmux send-keys -t "$SESSION" \
        "CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT} --alg PiCO-SC --C ${C} --k ${K}" \
        Enter
done

echo ""
echo "Attach:  tmux attach -t $SESSION"
echo "Kill:    tmux kill-session -t $SESSION"
