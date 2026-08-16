#!/usr/bin/env bash
# Runs the 6 algorithms (Fixed variant preferred where one exists) through
# the MAIN comparison pipeline (scripts/run_pipeline.py) -- NOT
# verify_scripts/ -- for C=20, k in {1,5,10,15,19}, spread across GPU slots.
# A failed (algorithm, k) cell is logged as a warning and does NOT stop the
# rest of the batch; once every cell in the queue has been attempted once,
# any that failed are automatically retried a second time.
#
# Algorithms used: CLPL, PRODEN (no Fixed variant exists for either -- the
# originals were already audited as paper-faithful), PiCO-Fixed,
# MCL-LOG-Fixed, ComCo-Fixed (Fixed preferred per request), SCL-NL (no
# Fixed variant).
#
# Each (algorithm, k) cell gets its own run_name (main_c20_k<k>_<alg>), so
# concurrent GPU workers never share a shard file -- same reasoning as
# run_paper_bench_parallel.sh's per-job run_name scheme.
#
# Usage:
#   scripts/run_main_pipeline_batch.sh                       # 8 GPUs (0-7), 200 epochs
#   scripts/run_main_pipeline_batch.sh --gpus 0,1,2,3          # fewer GPUs
#   scripts/run_main_pipeline_batch.sh --epochs 300            # override epoch count
#   scripts/run_main_pipeline_batch.sh --dry_run                # preview, run nothing

set -uo pipefail   # NOT -e: a failed cell must not kill the dispatcher

cd "$(dirname "$0")/.."

GPUS_CSV="0,1,2,3,4,5,6,7"
EPOCHS=200
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)    GPUS_CSV="$2"; shift 2 ;;
        --epochs)  EPOCHS="$2"; shift 2 ;;
        --dry_run) DRY_RUN=1; shift ;;
        *) echo "error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
NUM_GPUS=${#GPUS[@]}

ALGORITHMS=(CLPL PRODEN PiCO-Fixed MCL-LOG-Fixed ComCo-Fixed SCL-NL)
K_VALUES=(1 5 10 15 19)

LOG_DIR="logs/main_pipeline_batch"
mkdir -p "$LOG_DIR"
FAIL_LOG="$LOG_DIR/failures.log"
: > "$FAIL_LOG"

JOBS=()
for alg in "${ALGORITHMS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        JOBS+=("${alg}|${k}")
    done
done

_run_name_for() {
    # "PiCO-Fixed" -> "pico_fixed"
    echo "main_c20_k${2}_$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
}

echo "Jobs: ${#JOBS[@]}  (${#ALGORITHMS[@]} algorithms x ${#K_VALUES[@]} k-values), C=20, epochs=$EPOCHS"
echo "Algorithms: ${ALGORITHMS[*]}"
echo "k values: ${K_VALUES[*]}"
echo "GPU slots: ${GPUS[*]}"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    for job in "${JOBS[@]}"; do
        IFS='|' read -r alg k <<< "$job"
        echo "  alg=$alg k=$k -> run_name=$(_run_name_for "$alg" "$k")"
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
    # Runs inline in the main shell on purpose -- `wait` inside a `$(...)`
    # subshell can't reliably reap a job started in the parent shell (not
    # its OS-level parent); see run_paper_bench_parallel.sh's own fix for
    # the full explanation of this exact bug.
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
    # $1 = nameref to an array of "alg|k" job strings, $2 = pass label for logging.
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
        IFS='|' read -r alg k <<< "$job"
        gpu="${GPUS[$slot]}"
        run_name=$(_run_name_for "$alg" "$k")
        log_file="$LOG_DIR/${run_name}.log"
        desc="run_name=$run_name alg=$alg k=$k"

        echo "[START $((idx + 1))/$total][$pass_label] gpu=$gpu $desc"
        CUDA_VISIBLE_DEVICES="$gpu" python scripts/run_pipeline.py run \
            --run_name "$run_name" --algorithms "$alg" --c_values 20 --only_k "$k" --epochs "$EPOCHS" \
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
        IFS='|' read -r alg k <<< "$job"
        echo "  alg=$alg k=$k  (log: $LOG_DIR/$(_run_name_for "$alg" "$k").log)"
    done
else
    echo "Done. All ${#JOBS[@]} job(s) completed successfully$( [[ ${#PASS1_FAILED[@]} -gt 0 ]] && echo " (after retry)" )."
fi
echo ""
echo "Results: results/main_c20_k<k>_<alg>/results.csv"
