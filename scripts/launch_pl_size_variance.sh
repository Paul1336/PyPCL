#!/usr/bin/env bash
# Launch 4 tmux sessions for PL-size-variance experiment (GPU 0-3)
# Each GPU handles 1/4 of seeds; all 3 variance levels are run per seed.
# Usage: bash scripts/launch_pl_size_variance.sh

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="PyPCL"
SCRIPT="scripts/run_pl_size_variance.py"
NUM_GPUS=4

echo "Working dir: $WORKDIR"
echo "Conda env:   $CONDA_ENV"
echo ""

for GPU_ID in 0 1 2 3; do
    SESSION="plvar_gpu${GPU_ID}"

    tmux kill-session -t "$SESSION" 2>/dev/null
    tmux new-session -d -s "$SESSION"

    tmux send-keys -t "$SESSION" "cd \"$WORKDIR\"" Enter
    tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter
    tmux send-keys -t "$SESSION" \
        "CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT} --gpu_id ${GPU_ID} --num_gpus ${NUM_GPUS}" \
        Enter

    echo "  ${SESSION}  →  GPU ${GPU_ID}"
done

echo ""
echo "Attach:   tmux attach -t plvar_gpu0   (or gpu1 / gpu2 / gpu3)"
echo "List:     tmux ls"
echo "Kill all: for i in 0 1 2 3; do tmux kill-session -t plvar_gpu\$i; done"
