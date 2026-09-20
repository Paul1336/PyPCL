#!/usr/bin/env bash
# Waits for the ab_sweep_0915 orchestrator (GPUs 0-3) to finish, then launches
# the ema_sweep_0915 n-experiment reference backfill (PiCO-Fixed, k=10/15/19)
# on the SAME GPUs 0-3 only (per user: no other GPUs are available).
set -u
cd /nfs/home/Paul/PyPCL
mkdir -p logs
WAITLOG=logs/ema_sweep_0915.nref.wait.log
PY=/home/Paul/miniconda/envs/PyPCL/bin/python

echo "$(date -Iseconds) waiting for ab_sweep_0915 orchestrator to finish..." >> "$WAITLOG"
while pgrep -f "run_ab_sweep.py run --run_name ab_sweep_0915" > /dev/null; do
    sleep 60
done
echo "$(date -Iseconds) ab_sweep_0915 orchestrator no longer running -- launching ema_sweep_0915 n-reference backfill" >> "$WAITLOG"

nohup "$PY" scripts/run_ema_sweep.py run --run_name ema_sweep_0915 \
    --gpus 0 1 2 3 --slots_per_gpu 4 --experiments n --bases PiCO-Fixed \
    --epochs 200 --batch_size 512 --report_every 10 --poll 120 \
    > logs/ema_sweep_0915.orchestrator4.log 2>&1 &
echo "$(date -Iseconds) launched ema_sweep_0915 n-reference orchestrator pid $!" >> "$WAITLOG"
