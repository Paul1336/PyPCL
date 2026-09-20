#!/usr/bin/env bash
# Waits for the still-running ema_sweep_0915 orchestrator (GPUs 0-3) to
# finish, then launches the Factor A/B confidence-update-mechanism sweep
# (scripts/run_ab_sweep.py) on the same GPUs. Meant to be started once via
# nohup+disown so it survives the launching shell/session closing.
set -u
cd /nfs/home/Paul/PyPCL
mkdir -p logs
WAITLOG=logs/ab_sweep_0915.wait.log
PY=/home/Paul/miniconda/envs/PyPCL/bin/python

echo "$(date -Iseconds) waiting for ema_sweep_0915 orchestrator to finish..." >> "$WAITLOG"
while pgrep -f "run_ema_sweep.py run --run_name ema_sweep_0915" > /dev/null; do
    sleep 60
done
echo "$(date -Iseconds) ema_sweep_0915 orchestrator no longer running -- launching ab_sweep_0915" >> "$WAITLOG"

nohup "$PY" scripts/run_ab_sweep.py run --run_name ab_sweep_0915 \
    --gpus 0 1 2 3 --slots_per_gpu 4 --epochs 200 --batch_size 512 --report_every 10 --poll 120 \
    > logs/ab_sweep_0915.orchestrator.log 2>&1 &
echo "$(date -Iseconds) launched ab_sweep_0915 orchestrator pid $!" >> "$WAITLOG"
