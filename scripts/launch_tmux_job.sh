#!/usr/bin/env bash
# Opens one named tmux session pinned to one GPU and runs a single command
# in it. Lighter-weight than scripts/launch_multi_gpu.sh (which spawns one
# session PER GPU for a single --run_name's cooperative multi-worker run) --
# this is for launching one arbitrary, independent job by hand.
#
# On start, the session does:
#   cd <repo root>
#   conda activate <conda_env>
#   export CUDA_DEVICE_ORDER=PCI_BUS_ID
#   CUDA_VISIBLE_DEVICES=<gpu> <command>
#
# Usage:
#   scripts/launch_tmux_job.sh <session_name> <gpu_id> ["<command>"]
#   scripts/launch_tmux_job.sh <session_name> <gpu_id> --conda_env NAME ["<command>"]
#
# <command> is optional; if omitted it defaults to the SCL-NL/CIFAR-10 k=9
# example below. Wrap it in quotes if it has spaces (it always will).
#
# Examples:
#   scripts/launch_tmux_job.sh scl_nl_run 0
#   scripts/launch_tmux_job.sh proden_run 1 "python scripts/run_pipeline.py run --run_name proden_mnist --algorithms PRODEN --dataset mnist --epochs 200"
#   scripts/launch_tmux_job.sh test_run 2 --conda_env myenv "python -c 'import torch; print(torch.cuda.is_available())'"

set -euo pipefail

WORKDIR="$(cd "$(dirname "$0")/.." && pwd)"
CONDA_ENV="PyPCL"
DEFAULT_CMD="python scripts/run_pipeline.py run --run_name scl_nl_cifar10 --algorithms SCL-NL --dataset cifar10 --only_k 9 --epochs 500"

SESSION="${1:-}"
GPU="${2:-}"
shift 2 2>/dev/null || true

if [[ -z "$SESSION" || -z "$GPU" ]]; then
    echo "usage: $0 <session_name> <gpu_id> [--conda_env NAME] [\"<command>\"]" >&2
    exit 1
fi

CMD="$DEFAULT_CMD"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --conda_env) CONDA_ENV="$2"; shift 2 ;;
        *)           CMD="$1"; shift ;;
    esac
done

FULL_CMD="CUDA_VISIBLE_DEVICES=${GPU} ${CMD}"

echo "Session:    $SESSION"
echo "GPU:        $GPU"
echo "Conda env:  $CONDA_ENV"
echo "Workdir:    $WORKDIR"
echo "Command:    $FULL_CMD"
echo ""

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION"
tmux send-keys -t "$SESSION" "cd \"$WORKDIR\"" Enter
tmux send-keys -t "$SESSION" "conda activate $CONDA_ENV" Enter
tmux send-keys -t "$SESSION" "export CUDA_DEVICE_ORDER=PCI_BUS_ID" Enter
tmux send-keys -t "$SESSION" "$FULL_CMD" Enter

echo "Started. Attach with: tmux attach -t $SESSION"
echo "Kill with:            tmux kill-session -t $SESSION"
