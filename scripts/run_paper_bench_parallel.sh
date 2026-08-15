#!/usr/bin/env bash
# Runs every paper-original-benchmark experiment cell (see
# docs/00_paper_alignment_guide.md's "資料集支援" section and each
# docs/*_explanation.md's "實驗 Config" section) across N GPUs in parallel,
# one job per free GPU slot at a time.
#
# Unlike scripts/run_paper_original_benchmarks.sh (sequential, single GPU),
# this dispatches jobs onto whichever GPU slot frees up next. A FAILED job
# is logged to $FAIL_LOG and does NOT stop the rest of the queue -- the
# dispatcher just moves on to the next job.
#
# Each job gets its own run_name (paper__dataset), not a shared one, so
# concurrent jobs never write to the same results/<run_name>/shards/worker0.csv
# (every job uses --gpu_id 0 --num_gpus 1 internally; CUDA_VISIBLE_DEVICES is
# what actually pins it to a physical GPU -- see the design note at the
# bottom of this file for why run_name isolation, not --gpu_id, is used to
# avoid collisions here).
#
# Usage:
#   scripts/run_paper_bench_parallel.sh                 # use GPUs 0-3
#   scripts/run_paper_bench_parallel.sh 0 2 5 7          # explicit GPU ids
#   scripts/run_paper_bench_parallel.sh --dry_run        # print the queue, run nothing

set -uo pipefail   # NOT -e: a failed job must not kill the dispatcher

cd "$(dirname "$0")/.."

DRY_RUN=0
GPUS=()
for arg in "$@"; do
    if [[ "$arg" == "--dry_run" ]]; then
        DRY_RUN=1
    else
        GPUS+=("$arg")
    fi
done
[[ ${#GPUS[@]} -eq 0 ]] && GPUS=(0 1 2 3)
NUM_GPUS=${#GPUS[@]}

LOG_DIR="logs/paper_bench_parallel"
mkdir -p "$LOG_DIR"
FAIL_LOG="$LOG_DIR/failures.log"
SUMMARY_LOG="$LOG_DIR/summary.log"
: > "$FAIL_LOG"
: > "$SUMMARY_LOG"

# ─── Job queue ──────────────────────────────────────────────────────────────
# Format: "group|algorithms (space-separated)|dataset|extra run_pipeline.py args"
# `group` is only used to name run_name (group__dataset) and log files.
JOBS=(
    # --- CLPL (Cour et al., JMLR 2011) -- own UCI set: dermatology/ecoli/abalone ---
    "clpl|CLPL|clpl-lost|--epochs 100"
    "clpl|CLPL|clpl-fiw|--epochs 100"
    "clpl|CLPL|dermatology|--epochs 200"
    "clpl|CLPL|ecoli|--epochs 200"
    "clpl|CLPL|abalone|--epochs 200"

    # --- PRODEN (Lv et al., ICML 2020) ---
    "proden|PRODEN|mnist|--epochs 200 --batch_size 256"
    "proden|PRODEN|fashion-mnist|--epochs 200 --batch_size 256"
    "proden|PRODEN|kmnist|--epochs 200 --batch_size 256"
    "proden|PRODEN|yeast|--epochs 200"
    "proden|PRODEN|texture|--epochs 200"
    "proden|PRODEN|dermatology|--epochs 200"
    "proden|PRODEN|synthetic-control|--epochs 200"
    "proden|PRODEN|lost|--epochs 200"
    "proden|PRODEN|msrcv2|--epochs 200"
    "proden|PRODEN|birdsong|--epochs 200"
    "proden|PRODEN|soccer-player|--epochs 200"
    "proden|PRODEN|yahoo-news|--epochs 200"

    # --- MCL-LOG (Feng et al., ICML 2020) -- run original + Fixed side by side ---
    "mcl|MCL-LOG MCL-LOG-Fixed|mnist|--epochs 200"
    "mcl|MCL-LOG MCL-LOG-Fixed|fashion-mnist|--epochs 200"
    "mcl|MCL-LOG MCL-LOG-Fixed|kmnist|--epochs 200"
    "mcl|MCL-LOG MCL-LOG-Fixed|20newsgroups|--epochs 100"
    "mcl|MCL-LOG MCL-LOG-Fixed|yeast|--epochs 200"
    "mcl|MCL-LOG MCL-LOG-Fixed|texture|--epochs 200"
    "mcl|MCL-LOG MCL-LOG-Fixed|dermatology|--epochs 200"
    "mcl|MCL-LOG MCL-LOG-Fixed|synthetic-control|--epochs 200"

    # --- SCL-NL (Chou et al., ICML 2020) ---
    "scl_nl|SCL-NL|mnist|--epochs 300"
    "scl_nl|SCL-NL|kmnist|--epochs 300"
    "scl_nl|SCL-NL|fashion-mnist|--epochs 300"

    # --- PiCO (Wang et al., ICLR 2022) -- run original + Fixed side by side ---
    "pico|PiCO PiCO-Fixed|cub200|--epochs 100"
    "pico|PiCO PiCO-Fixed|cifar100-h|--c_values 100 --epochs 200"

    # --- ComCo (Jiang et al., Neural Networks 2024) -- run original + Fixed side by side ---
    "comco|ComCo ComCo-Fixed|cub200|--epochs 100"
    # sun397 is UNVERIFIED (17-37GB, never downloaded in the original pass) --
    # kept as a small smoke test, not a full run. Remove this line once
    # you've confirmed it works and want a real-scale run instead.
    "comco|ComCo|sun397|--epochs 1 --batch_size 16"
)

TOTAL=${#JOBS[@]}
echo "Queue: $TOTAL jobs across ${NUM_GPUS} GPU(s): ${GPUS[*]}"
echo "Logs:  $LOG_DIR/<group>__<dataset>.log"
echo "Failures logged to: $FAIL_LOG"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    for job in "${JOBS[@]}"; do
        IFS='|' read -r group algos dataset extra <<< "$job"
        echo "  run_name=${group}__${dataset}  dataset=$dataset  algorithms=[$algos]  $extra"
    done
    echo ""
    echo "(dry run — no jobs launched)"
    exit 0
fi

declare -a SLOT_PID
declare -a SLOT_DESC
declare -a SLOT_LOG
for ((s = 0; s < NUM_GPUS; s++)); do
    SLOT_PID[s]=0
    SLOT_DESC[s]=""
    SLOT_LOG[s]=""
done

_reap_slot() {
    # $1 = slot index. Waits on that slot's PID (already known finished),
    # records pass/fail, and clears the slot.
    local s="$1"
    local pid="${SLOT_PID[$s]}"
    wait "$pid"
    local code=$?
    if [[ $code -ne 0 ]]; then
        echo "[FAIL] gpu=${GPUS[$s]} exit=$code :: ${SLOT_DESC[$s]}  (log: ${SLOT_LOG[$s]})" | tee -a "$FAIL_LOG"
    else
        echo "[OK]   gpu=${GPUS[$s]} :: ${SLOT_DESC[$s]}" | tee -a "$SUMMARY_LOG"
    fi
    SLOT_PID[$s]=0
}

_find_free_slot() {
    # Prints a free slot index to stdout, blocking (polling) until one
    # exists. Reaps any slot whose process has already finished.
    while true; do
        for ((s = 0; s < NUM_GPUS; s++)); do
            local pid="${SLOT_PID[$s]}"
            if [[ "$pid" -eq 0 ]]; then
                echo "$s"
                return
            fi
            if ! kill -0 "$pid" 2>/dev/null; then
                _reap_slot "$s" >&2
                echo "$s"
                return
            fi
        done
        sleep 5
    done
}

job_idx=0
while [[ $job_idx -lt $TOTAL ]]; do
    slot=$(_find_free_slot)
    IFS='|' read -r group algos dataset extra <<< "${JOBS[$job_idx]}"
    gpu="${GPUS[$slot]}"
    run_name="${group}__${dataset}"
    log_file="$LOG_DIR/${run_name}.log"
    desc="run_name=$run_name dataset=$dataset algorithms=[$algos] $extra"

    echo "[START $((job_idx + 1))/$TOTAL] gpu=$gpu :: $desc"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_pipeline.py run \
        --run_name "$run_name" --algorithms $algos --dataset "$dataset" $extra \
        > "$log_file" 2>&1 &

    SLOT_PID[$slot]=$!
    SLOT_DESC[$slot]="$desc"
    SLOT_LOG[$slot]="$log_file"
    job_idx=$((job_idx + 1))
done

# Drain remaining running jobs.
for ((s = 0; s < NUM_GPUS; s++)); do
    if [[ "${SLOT_PID[$s]}" -ne 0 ]]; then
        _reap_slot "$s"
    fi
done

echo ""
if [[ -s "$FAIL_LOG" ]]; then
    n_fail=$(wc -l < "$FAIL_LOG")
    echo "Done. ${n_fail} job(s) FAILED -- see $FAIL_LOG:"
    cat "$FAIL_LOG"
else
    echo "Done. All $TOTAL jobs completed successfully."
fi

# ─── Design note: why per-job run_name instead of shared run_name + --gpu_id ───
# src/pipeline/results.py's shard file is results/<run_name>/shards/worker<gpu_id>.csv.
# scripts/launch_multi_gpu.sh's pattern (shared run_name, distinct --gpu_id per
# worker, --num_gpus = worker count) is correct when N workers cooperatively
# split ONE --algorithms list across ONE dataset (gpu.assign_algorithms does
# `i % num_gpus == gpu_id` round-robin). This script instead parallelizes
# across DIFFERENT (dataset, algorithms) cells -- forcing those onto a shared
# run_name with hand-picked --gpu_id values would require --num_gpus to match
# too, which would then incorrectly re-split each job's own --algorithms list
# via that same round-robin. Giving every job its own run_name sidesteps this
# entirely: every job safely uses the default --gpu_id 0 --num_gpus 1 (so its
# full --algorithms list runs intact), and there is no shared shard file for
# concurrent jobs to race on.
