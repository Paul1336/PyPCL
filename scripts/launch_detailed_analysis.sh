#!/usr/bin/env bash
# Launch 4 tmux sessions for detailed analysis (GPU 0-3).
# Each GPU trains one algorithm across k=5, 10, 15 sequentially.
#
# Usage: bash scripts/launch_detailed_analysis.sh

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="PyPCL"
SCRIPT="scripts/run_detailed_analysis.py"
C=20

echo "Working dir: $WORKDIR"
echo "Conda env:   $CONDA_ENV"
echo ""

declare -A GPU_ALG
GPU_ALG[0]="PiCO"
GPU_ALG[1]="PiCO-CLS"
GPU_ALG[2]="PiCO-SC"
GPU_ALG[3]="ComCo"

for GPU_ID in 0 1 2 3; do
    ALG="${GPU_ALG[$GPU_ID]}"
    SESSION="detail_gpu${GPU_ID}"

    tmux kill-session -t "$SESSION" 2>/dev/null
    tmux new-session -d -s "$SESSION"

    tmux send-keys -t "$SESSION" "cd \"$WORKDIR\"" Enter
    tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter

    for K in 5 10 15 6 7 8 9 11 12 13 14; do
        tmux send-keys -t "$SESSION" \
            "CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT} --alg ${ALG} --C ${C} --k ${K}" \
            Enter
    done

    echo "  ${SESSION}  →  GPU ${GPU_ID}  alg=${ALG}  k=5,10,15 then 6,7,8,9,11,12,13,14"
done

echo ""
echo "Attach:   tmux attach -t detail_gpu0   (or gpu1 / gpu2 / gpu3)"
echo "List:     tmux ls"
echo "Kill all: for i in 0 1 2 3; do tmux kill-session -t detail_gpu\$i; done"
