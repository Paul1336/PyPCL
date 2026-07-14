#!/usr/bin/env bash
# Launch tmux sessions for GPU 1-3 (Cour2011, PRODEN, PiCO)
# Run from anywhere inside the PyPCL repo:
#   bash scripts/launch_gpu_0to3.sh

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="PyPCL"
SCRIPT="scripts/run_adam_comparison.py"

echo "Working dir: $WORKDIR"
echo "Conda env:   $CONDA_ENV"
echo ""

for IDX in 0 1 2; do
    CUDA_GPU=$((IDX + 1))
    SESSION="c20_gpu${CUDA_GPU}"

    tmux kill-session -t "$SESSION" 2>/dev/null
    tmux new-session -d -s "$SESSION"

    tmux send-keys -t "$SESSION" "cd \"$WORKDIR\"" Enter
    tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter
    tmux send-keys -t "$SESSION" "CUDA_VISIBLE_DEVICES=${CUDA_GPU} python ${SCRIPT} --gpu_id ${IDX} --num_gpus 7" Enter

    echo "  ${SESSION}  →  physical GPU ${CUDA_GPU}  (algo idx ${IDX})"
done

echo ""
echo "Attach:   tmux attach -t c20_gpu1   (or c20_gpu2 / c20_gpu3)"
echo "List:     tmux ls"
echo "Kill all: for i in 1 2 3; do tmux kill-session -t c20_gpu\$i; done"
