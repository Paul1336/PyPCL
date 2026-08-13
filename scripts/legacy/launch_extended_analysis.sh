#!/usr/bin/env bash
# Launch 5 tmux sessions for extended analysis (GPU 0-4).
# Methods: PiCO / PiCO-Uniform / PiCO-CLS / PiCO-SC / ComCo
# k values: 5  10  15  19
# Epochs:   500
#
# Usage: bash scripts/launch_extended_analysis.sh

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="PyPCL"
SCRIPT="scripts/run_extended_analysis.py"
C=20
EPOCHS=500

echo "Working dir: $WORKDIR"
echo "Conda env:   $CONDA_ENV"
echo ""

declare -A GPU_ALG
GPU_ALG[0]="PiCO"
GPU_ALG[1]="PiCO-Uniform"
GPU_ALG[2]="PiCO-CLS"
GPU_ALG[3]="PiCO-SC"
GPU_ALG[4]="ComCo"

for GPU_ID in 0 1 2 3 4; do
    ALG="${GPU_ALG[$GPU_ID]}"
    SESSION="ext_gpu${GPU_ID}"

    tmux kill-session -t "$SESSION" 2>/dev/null
    tmux new-session -d -s "$SESSION"

    tmux send-keys -t "$SESSION" "cd \"$WORKDIR\"" Enter
    tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter

    for K in 5 10 15 19; do
        tmux send-keys -t "$SESSION" \
            "CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT} --alg ${ALG} --C ${C} --k ${K} --epochs ${EPOCHS}" \
            Enter
    done

    echo "  ${SESSION}  →  GPU ${GPU_ID}  alg=${ALG}  k=5,10,15,19"
done

echo ""
echo "Attach:   tmux attach -t ext_gpu0   (or ext_gpu1 .. ext_gpu4)"
echo "List:     tmux ls"
echo "Kill all: for i in 0 1 2 3 4; do tmux kill-session -t ext_gpu\$i; done"
