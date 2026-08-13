#!/usr/bin/env bash
# Generic multi-GPU launcher for scripts/run_pipeline.py.
#
# Given how many GPUs (and optionally which physical ids) you want to use,
# spawns one tmux session per GPU, each running:
#
#   CUDA_VISIBLE_DEVICES=<physical_id> python scripts/run_pipeline.py run \
#       --run_name <run_name> --gpu_id <worker_index> --num_gpus <worker_count> <your extra args>
#
# The pipeline's own round-robin (gpu.assign_algorithms) then splits
# --algorithms across the workers automatically.
#
# Usage:
#   scripts/launch_multi_gpu.sh --run_name NAME --gpus N [run_pipeline.py run args...]
#   scripts/launch_multi_gpu.sh --run_name NAME --gpu_ids 0,2,5 [run_pipeline.py run args...]
#
# Options (consumed by this script; everything else is forwarded as-is to
# `python scripts/run_pipeline.py run`):
#   --run_name NAME       Required. Passed straight through as --run_name.
#   --gpus N               Use N GPUs, physical ids 0..N-1 (default numbering).
#   --gpu_ids "0,2,5"       Explicit physical GPU ids (comma-separated). Overrides --gpus.
#   --conda_env NAME        Conda environment to activate in each session (default: PyPCL).
#   --session_prefix NAME   tmux session name prefix (default: run_name).
#   --dry_run                Print the commands and tmux session names without launching anything.
#
# Examples:
#   # 8 GPUs (ids 0-7), all 14 algorithms, default schedule
#   scripts/launch_multi_gpu.sh --run_name full --gpus 8 --c_values 5 20 --epochs 200
#
#   # Only physical GPUs 2, 5, 7 (e.g. shared box), 5 specific algorithms
#   scripts/launch_multi_gpu.sh --run_name full --gpu_ids 2,5,7 \
#       --algorithms CLPL PRODEN MCL-LOG PiCO ComCo --c_values 5 20
#
#   # Preview the commands first
#   scripts/launch_multi_gpu.sh --run_name full --gpus 4 --epochs 200 --dry_run

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="PyPCL"
SESSION_PREFIX=""
RUN_NAME=""
NUM_GPUS=""
GPU_IDS_CSV=""
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_name)       RUN_NAME="$2"; shift 2 ;;
        --gpus)           NUM_GPUS="$2"; shift 2 ;;
        --gpu_ids)        GPU_IDS_CSV="$2"; shift 2 ;;
        --conda_env)      CONDA_ENV="$2"; shift 2 ;;
        --session_prefix) SESSION_PREFIX="$2"; shift 2 ;;
        --dry_run)        DRY_RUN=1; shift ;;
        *)                EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$RUN_NAME" ]]; then
    echo "error: --run_name is required" >&2
    exit 1
fi

if [[ -n "$GPU_IDS_CSV" ]]; then
    IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_CSV"
elif [[ -n "$NUM_GPUS" ]]; then
    GPU_IDS=()
    for ((i = 0; i < NUM_GPUS; i++)); do GPU_IDS+=("$i"); done
else
    echo "error: pass either --gpus N or --gpu_ids 0,1,2,..." >&2
    exit 1
fi

NUM_WORKERS=${#GPU_IDS[@]}
[[ -z "$SESSION_PREFIX" ]] && SESSION_PREFIX="$RUN_NAME"

echo "Working dir:  $WORKDIR"
echo "Conda env:    $CONDA_ENV"
echo "Run name:     $RUN_NAME"
echo "Workers:      $NUM_WORKERS  (physical GPUs: ${GPU_IDS[*]})"
echo "Extra args:   ${EXTRA_ARGS[*]:-<none>}"
echo ""

for i in "${!GPU_IDS[@]}"; do
    PHYS_GPU="${GPU_IDS[$i]}"
    SESSION="${SESSION_PREFIX}_gpu${PHYS_GPU}"
    CMD="CUDA_VISIBLE_DEVICES=${PHYS_GPU} python scripts/run_pipeline.py run --run_name ${RUN_NAME} --gpu_id ${i} --num_gpus ${NUM_WORKERS} ${EXTRA_ARGS[*]:-}"

    echo "  ${SESSION}  ->  physical GPU ${PHYS_GPU}  (worker ${i}/${NUM_WORKERS})"
    echo "    ${CMD}"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        continue
    fi

    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION"
    tmux send-keys -t "$SESSION" "cd \"$WORKDIR\"" Enter
    tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter
    tmux send-keys -t "$SESSION" "$CMD" Enter
done

echo ""
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "(dry run — no tmux sessions started)"
else
    echo "Attach:   tmux attach -t ${SESSION_PREFIX}_gpu${GPU_IDS[0]}   (etc., one per physical GPU id above)"
    echo "List:     tmux ls"
    echo "Kill all: for g in ${GPU_IDS[*]}; do tmux kill-session -t ${SESSION_PREFIX}_gpu\$g; done"
fi
