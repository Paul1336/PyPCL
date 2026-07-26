#!/usr/bin/env bash
# Launch 3 tmux sessions for OP / OP-W / CPE comparison (GPU 4, 5, 6)
# Usage: bash scripts/launch_op_cpe.sh

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="PyPCL"
SCRIPT="scripts/run_op_cpe_comparison.py"
NUM_GPUS=3

# Physical GPU IDs → logical gpu_id mapping:
#   CUDA_VISIBLE_DEVICES=4  --gpu_id 0  →  OP
#   CUDA_VISIBLE_DEVICES=5  --gpu_id 1  →  OP-W
#   CUDA_VISIBLE_DEVICES=6  --gpu_id 2  →  CPE
PHYS_GPUS=(4 5 6)

echo "Working dir: $WORKDIR"
echo "Conda env:   $CONDA_ENV"
echo ""

for LOGICAL_ID in 0 1 2; do
    PHYS_GPU="${PHYS_GPUS[$LOGICAL_ID]}"
    SESSION="op_cpe_gpu${PHYS_GPU}"

    tmux kill-session -t "$SESSION" 2>/dev/null
    tmux new-session -d -s "$SESSION"

    tmux send-keys -t "$SESSION" "cd \"$WORKDIR\"" Enter
    tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter
    tmux send-keys -t "$SESSION" \
        "CUDA_VISIBLE_DEVICES=${PHYS_GPU} python ${SCRIPT} --gpu_id ${LOGICAL_ID} --num_gpus ${NUM_GPUS}" \
        Enter

    ALGO_NAME=("OP" "OP-W" "CPE")
    echo "  ${SESSION}  →  GPU ${PHYS_GPU}  (${ALGO_NAME[$LOGICAL_ID]})"
done

echo ""
echo "Attach:   tmux attach -t op_cpe_gpu4   (or gpu5 / gpu6)"
echo "List:     tmux ls"
echo "Kill all: for i in 4 5 6; do tmux kill-session -t op_cpe_gpu\$i; done"
