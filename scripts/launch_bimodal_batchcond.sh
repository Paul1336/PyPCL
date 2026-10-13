#!/usr/bin/env bash
# Bimodal-k BatchCond validation (2026-10-13): C=20, each sample k=5 or k=15
# w.p. 0.5 (mean 10 = BATCH_COND_THRESHOLD). 2 batchings x 5 algorithms x 3
# seeds = 30 runs, queued onto ONE GPU with at most $SLOTS concurrent.
# Usage: scripts/launch_bimodal_batchcond.sh [GPU] [SLOTS]
set -u
GPU=${1:-3}; SLOTS=${2:-4}
PY=/home/Paul/miniconda/envs/PyPCL/bin/python
DATA=/nfs/home/Paul/PyPCL/data
ALGOS="ComCo-Weighted-A000 ComCo-Weighted-A100 ComCo-Weighted-BatchCond ComCo-Weighted-BatchRandom ComCo-Weighted-SampleCond"
mkdir -p logs/bimodal_1013
jobs=()
for seed in 42 43 44; do
  for batching in bucketed random; do
    wid=0
    for a in $ALGOS; do
      for s in 42 43 44; do [ "$s" = "$seed" ] && w=$wid; wid=$((wid+1)); done
      jobs+=("$batching $a $seed $w")
    done
  done
done
printf '%s\n' "${jobs[@]}" | xargs -P "$SLOTS" -L 1 bash -c '
  batching=$0; algo=$1; seed=$2; wid=$3
  run=bimodal5-15_${batching}_1013
  log=logs/bimodal_1013/${run}_${algo}_s${seed}.log
  echo "$(date +%F_%T) START $run $algo seed=$seed" >> logs/bimodal_1013/launcher.log
  CUDA_VISIBLE_DEVICES='"$GPU"' '"$PY"' scripts/run_pipeline.py run --run_name $run \
    --algorithms ComCo-Weighted-A000 ComCo-Weighted-A100 ComCo-Weighted-BatchCond ComCo-Weighted-BatchRandom ComCo-Weighted-SampleCond \
    --algo $algo --gpu_id $wid --num_gpus 15 --c_values 20 --seeds $seed --epochs 200 \
    --only_bimodal 5 15 0.5 --batching $batching --data_dir '"$DATA"' > $log 2>&1
  echo "$(date +%F_%T) END rc=$? $run $algo seed=$seed" >> logs/bimodal_1013/launcher.log
'
echo "$(date +%F_%T) ALL DONE" >> logs/bimodal_1013/launcher.log
