#!/usr/bin/env bash
# PiCO-Oracle graduated-correction threshold sweep: for each k in {19,15,10}
# (C=20 fixed) and each precision threshold in THRESHOLDS, trains PiCO-Oracle
# once with --detail (per-class accuracy/loss + the before/after correction
# precision log; see src/pipeline/detail.py::train_pico_oracle_graded_epoch_with_stats).
#
# threshold is read from config.yaml's pico.oracle_precision_threshold, and
# `run_pipeline.py run --config <path>` accepts a config path per invocation
# -- so each threshold gets its own generated config file (a full copy of
# config.yaml with just that one key overridden) instead of editing the
# shared config.yaml, which would race across concurrently-running jobs.
#
# run_name / results dir get a "thresholdoracle_" prefix, distinct from the
# earlier "new_main_c20_*" batch already sitting in results/.
#
# Same GPU-parallel dispatcher w/ retry-after-full-pass pattern as
# scripts/run_main_pipeline_batch.sh (including the CUDA_DEVICE_ORDER fix --
# see that script's comments for the full rationale on both).
#
# Usage:
#   scripts/run_threshold_oracle_batch.sh                      # GPUs 0-3, 1 job/GPU, 200 epochs
#   scripts/run_threshold_oracle_batch.sh --jobs_per_gpu 2      # 2 jobs/GPU
#   scripts/run_threshold_oracle_batch.sh --dry_run             # preview, run nothing

set -uo pipefail   # NOT -e: a failed cell must not kill the dispatcher

cd "$(dirname "$0")/.."

# This server's nvidia-smi GPU numbering doesn't match CUDA's default
# ordering -- see scripts/run_main_pipeline_batch.sh's own comment.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

GPUS_CSV="0,1,2,3"
EPOCHS=200
JOBS_PER_GPU=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)          GPUS_CSV="$2"; shift 2 ;;
        --epochs)        EPOCHS="$2"; shift 2 ;;
        --jobs_per_gpu)  JOBS_PER_GPU="$2"; shift 2 ;;
        --dry_run)       DRY_RUN=1; shift ;;
        *) echo "error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

IFS=',' read -r -a PHYS_GPUS <<< "$GPUS_CSV"
GPUS=()
for g in "${PHYS_GPUS[@]}"; do
    for ((i = 0; i < JOBS_PER_GPU; i++)); do
        GPUS+=("$g")
    done
done
NUM_GPUS=${#GPUS[@]}

K_VALUES=(19 15 10)
THRESHOLDS=(0 0.05 0.07 0.09 0.11 0.15 0.2 0.25 0.5 0.75 1)

CONFIG_DIR="configs/thresholdoracle"
mkdir -p "$CONFIG_DIR"
LOG_DIR="logs/thresholdoracle_batch"
mkdir -p "$LOG_DIR"
FAIL_LOG="$LOG_DIR/failures.log"
: > "$FAIL_LOG"

_tag_for_threshold() {
    # 0.05 -> 0p05 ; 1 -> 1 ; 0 -> 0  (filesystem/run_name-safe)
    echo "$1" | sed 's/\./p/'
}

_run_name_for() {
    echo "thresholdoracle_c20_k${1}_t$(_tag_for_threshold "$2")"
}

_config_for_threshold() {
    # Idempotent: only generates the file the first time a given threshold
    # is requested, so concurrent jobs sharing the same threshold never race
    # to write it (all thresholds are pre-generated up front below anyway).
    local thr="$1"
    local tag
    tag=$(_tag_for_threshold "$thr")
    local path="$CONFIG_DIR/threshold_t${tag}.yaml"
    if [[ ! -f "$path" ]]; then
        python3 -c "
import yaml
with open('config.yaml', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('pico', {})['oracle_precision_threshold'] = float('$thr')
with open('$path', 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
"
    fi
    echo "$path"
}

for thr in "${THRESHOLDS[@]}"; do
    _config_for_threshold "$thr" > /dev/null
done

JOBS=()
for k in "${K_VALUES[@]}"; do
    for thr in "${THRESHOLDS[@]}"; do
        JOBS+=("${k}|${thr}")
    done
done

echo "Jobs: ${#JOBS[@]}  (${#K_VALUES[@]} k-values x ${#THRESHOLDS[@]} thresholds), C=20, epochs=$EPOCHS"
echo "k values: ${K_VALUES[*]}"
echo "thresholds: ${THRESHOLDS[*]}"
echo "Physical GPUs: ${PHYS_GPUS[*]}  (x${JOBS_PER_GPU} jobs/GPU -> ${NUM_GPUS} concurrent slot(s): ${GPUS[*]})"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    for job in "${JOBS[@]}"; do
        IFS='|' read -r k thr <<< "$job"
        echo "  k=$k threshold=$thr -> run_name=$(_run_name_for "$k" "$thr")  config=$(_config_for_threshold "$thr")"
    done
    echo ""
    echo "(dry run -- no jobs launched)"
    exit 0
fi

declare -a SLOT_PID SLOT_DESC SLOT_LOG SLOT_JOB

_reap_slot() {
    local s="$1"
    local pass_label="$2"
    wait "${SLOT_PID[$s]}"
    local code=$?
    if [[ $code -ne 0 ]]; then
        echo "[FAIL][$pass_label] gpu=${GPUS[$s]} ${SLOT_DESC[$s]} exit=$code  (log: ${SLOT_LOG[$s]})" | tee -a "$FAIL_LOG"
        FAILED_THIS_PASS+=("${SLOT_JOB[$s]}")
    else
        echo "[OK]  [$pass_label] gpu=${GPUS[$s]} ${SLOT_DESC[$s]}"
    fi
    SLOT_PID[$s]=0
}

FREE_SLOT=-1
_find_free_slot() {
    # Runs inline in the main shell on purpose -- see
    # run_main_pipeline_batch.sh's comment for why `wait` can't live inside
    # a $(...) subshell here.
    local pass_label="$1"
    while true; do
        for ((s = 0; s < NUM_GPUS; s++)); do
            if [[ "${SLOT_PID[$s]:-0}" -eq 0 ]]; then
                FREE_SLOT=$s
                return
            fi
        done
        wait -n
        for ((s = 0; s < NUM_GPUS; s++)); do
            pid="${SLOT_PID[$s]}"
            if [[ "$pid" -ne 0 ]] && ! kill -0 "$pid" 2>/dev/null; then
                _reap_slot "$s" "$pass_label"
            fi
        done
    done
}

_run_queue() {
    # $1 = nameref to an array of "k|threshold" job strings, $2 = pass label.
    local -n queue_ref=$1
    local pass_label="$2"
    local total=${#queue_ref[@]}
    local idx=0
    FAILED_THIS_PASS=()
    for ((s = 0; s < NUM_GPUS; s++)); do SLOT_PID[s]=0; done

    while [[ $idx -lt $total ]]; do
        _find_free_slot "$pass_label"
        slot=$FREE_SLOT
        job="${queue_ref[$idx]}"
        IFS='|' read -r k thr <<< "$job"
        gpu="${GPUS[$slot]}"
        run_name=$(_run_name_for "$k" "$thr")
        config_path=$(_config_for_threshold "$thr")
        log_file="$LOG_DIR/${run_name}.log"
        desc="run_name=$run_name k=$k threshold=$thr"

        echo "[START $((idx + 1))/$total][$pass_label] gpu=$gpu $desc"
        CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_pipeline.py run \
            --run_name "$run_name" --algorithms PiCO-Oracle --c_values 20 --only_k "$k" \
            --epochs "$EPOCHS" --detail --config "$config_path" \
            > "$log_file" 2>&1 &

        SLOT_PID[$slot]=$!
        SLOT_DESC[$slot]="$desc"
        SLOT_LOG[$slot]="$log_file"
        SLOT_JOB[$slot]="$job"
        idx=$((idx + 1))
    done

    for ((s = 0; s < NUM_GPUS; s++)); do
        if [[ "${SLOT_PID[$s]}" -ne 0 ]]; then
            _reap_slot "$s" "$pass_label"
        fi
    done
}

echo "=== Pass 1/2: ${#JOBS[@]} job(s) ==="
_run_queue JOBS "pass1"
PASS1_FAILED=("${FAILED_THIS_PASS[@]}")

if [[ ${#PASS1_FAILED[@]} -gt 0 ]]; then
    echo ""
    echo "=== Pass 1 done. ${#PASS1_FAILED[@]} job(s) failed -- retrying once ==="
    for job in "${PASS1_FAILED[@]}"; do echo "  retry: $job"; done
    echo ""
    RETRY_JOBS=("${PASS1_FAILED[@]}")
    _run_queue RETRY_JOBS "retry"
    FINAL_FAILED=("${FAILED_THIS_PASS[@]}")
else
    FINAL_FAILED=()
fi

echo ""
if [[ ${#FINAL_FAILED[@]} -gt 0 ]]; then
    echo "Done. ${#FINAL_FAILED[@]} job(s) still failing after retry -- see $FAIL_LOG:"
    for job in "${FINAL_FAILED[@]}"; do
        IFS='|' read -r k thr <<< "$job"
        echo "  k=$k threshold=$thr  (log: $LOG_DIR/$(_run_name_for "$k" "$thr").log)"
    done
else
    echo "Done. All ${#JOBS[@]} job(s) completed successfully$( [[ ${#PASS1_FAILED[@]} -gt 0 ]] && echo " (after retry)" )."
fi
echo ""
echo "Results: results/thresholdoracle_c20_k<k>_t<threshold>/results.csv"
echo "Detail output: results/thresholdoracle_c20_k<k>_t<threshold>/detail/PiCO-Oracle/C20_k<k>/"
