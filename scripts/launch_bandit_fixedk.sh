#!/usr/bin/env bash
# Paired-lookahead bandit vs fixed alpha (2026-10-13): C=20, fixed k in
# {5,10,15}, 10% held-out for the label-free reward. 3 k x 4 algorithms x 3
# seeds = 36 runs over GPUs ${GPUS} with SLOTS concurrent per GPU.
# Usage: scripts/launch_bandit_fixedk.sh ["0 1 2"] [SLOTS]
set -u
GPUS=(${1:-0 1 2}); SLOTS=${2:-4}
PY=/home/Paul/miniconda/envs/PyPCL/bin/python
DATA=/nfs/home/Paul/PyPCL/data
RUN=bandit_fixedk_1013
ALGOS=(ComCo-Weighted-BanditPaired ComCo-Weighted-BanditEpoch ComCo-Weighted-A000 ComCo-Weighted-A100)
mkdir -p logs/$RUN
jobs=(); n=0
for seed in 42 43 44; do for k in 15 10 5; do for a in "${ALGOS[@]}"; do
  jobs+=("$a $k $seed $n"); n=$((n+1))
done; done; done
TOTAL_SLOTS=$(( ${#GPUS[@]} * SLOTS ))
# slot s always runs on GPU GPUS[s % #GPUS]; each slot drains its own share of the queue
for s in $(seq 0 $((TOTAL_SLOTS-1))); do
  (
    gpu=${GPUS[$((s % ${#GPUS[@]}))]}
    for ((j=s; j<${#jobs[@]}; j+=TOTAL_SLOTS)); do
      read -r algo k seed wid <<< "${jobs[$j]}"
      log=logs/$RUN/${algo}_k${k}_s${seed}.log
      echo "$(date +%F_%T) START gpu$gpu $algo k=$k seed=$seed" >> logs/$RUN/launcher.log
      CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_pipeline.py run --run_name $RUN \
        --algorithms "${ALGOS[@]}" --algo $algo --gpu_id $wid --num_gpus ${#jobs[@]} \
        --c_values 20 --only_k $k --seeds $seed --epochs 200 --holdout_frac 0.1 \
        --data_dir $DATA > $log 2>&1
      echo "$(date +%F_%T) END rc=$? gpu$gpu $algo k=$k seed=$seed" >> logs/$RUN/launcher.log
    done
  ) &
done
wait
echo "$(date +%F_%T) ALL DONE" >> logs/$RUN/launcher.log
