#!/usr/bin/env bash
# Unified launcher for the six paper-exact verify_scripts/*_verify.py
# scripts. Lets you pick which papers to run and how many/which GPUs to
# spread them across -- each selected paper runs to completion on one GPU
# slot, next paper in the queue takes the slot when it frees up.
#
# Each script writes to its OWN results file (verify_results/<id>.csv), so
# unlike run_pipeline.py's multi-worker sharding there's no collision risk
# to design around here -- this is a much simpler queue than
# run_paper_bench_parallel.sh, just reusing the same dispatcher core (see
# that script for why `wait` must run inline in the main shell, never inside
# a `$(...)` command-substitution subshell -- a subshell isn't the OS-level
# parent of a background job started in the parent shell, so `wait` there
# can't reliably reap it; this script was written from the start avoiding
# that bug rather than fixing it after the fact).
#
# Usage:
#   verify_scripts/run_verify.sh                        # run all 6, GPUs 0-3
#   verify_scripts/run_verify.sh --methods pico comco    # just these two
#   verify_scripts/run_verify.sh --gpus 0,1               # only 2 GPU slots
#   verify_scripts/run_verify.sh --dry_run                # preview, run nothing
#   verify_scripts/run_verify.sh --extra-args pico "--epochs 10"   # smoke-test override
#
# Options:
#   --methods NAME [NAME ...]   Subset of: clpl proden pico mcl_log comco scl_nl
#                                 (default: all six)
#   --gpus "0,1,2,3"             Physical GPU ids to use as slots (default: 0,1,2,3).
#                                 CLPL is CPU-only (sklearn SVM) and still gets a
#                                 slot in the rotation, but CUDA_VISIBLE_DEVICES
#                                 for it is harmless/unused.
#   --extra-args METHOD "ARGS"   Append extra CLI args to one method's invocation
#                                 (repeatable). Useful for quick smoke runs, e.g.
#                                 --extra-args proden "--epochs 5 --batch_size 32".
#   --dry_run                    Print the planned dispatch, launch nothing.

set -uo pipefail   # NOT -e: one method failing must not kill the dispatcher

cd "$(dirname "$0")/.."

declare -A SCRIPT_OF=(
    [clpl]="verify_scripts/clpl_verify.py"
    [proden]="verify_scripts/proden_verify.py"
    [pico]="verify_scripts/pico_verify.py"
    [mcl_log]="verify_scripts/mcl_log_verify.py"
    [comco]="verify_scripts/comco_verify.py"
    [scl_nl]="verify_scripts/scl_nl_verify.py"
)
ALL_METHODS=(clpl proden pico mcl_log comco scl_nl)

METHODS=()
GPUS_CSV="0,1,2,3"
DRY_RUN=0
declare -A EXTRA_ARGS

while [[ $# -gt 0 ]]; do
    case "$1" in
        --methods)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                METHODS+=("$1"); shift
            done
            ;;
        --gpus)
            GPUS_CSV="$2"; shift 2 ;;
        --extra-args)
            EXTRA_ARGS["$2"]="$3"; shift 3 ;;
        --dry_run)
            DRY_RUN=1; shift ;;
        *)
            echo "error: unknown argument '$1'" >&2
            exit 1 ;;
    esac
done

[[ ${#METHODS[@]} -eq 0 ]] && METHODS=("${ALL_METHODS[@]}")
for m in "${METHODS[@]}"; do
    if [[ -z "${SCRIPT_OF[$m]+x}" ]]; then
        echo "error: unknown method '$m' (choices: ${ALL_METHODS[*]})" >&2
        exit 1
    fi
done

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
NUM_GPUS=${#GPUS[@]}

LOG_DIR="logs/verify_scripts"
mkdir -p "$LOG_DIR" verify_results
FAIL_LOG="$LOG_DIR/failures.log"
: > "$FAIL_LOG"

echo "Methods: ${METHODS[*]}"
echo "GPU slots: ${GPUS[*]}"
echo "Logs: $LOG_DIR/<method>.log"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    for m in "${METHODS[@]}"; do
        echo "  $m -> ${SCRIPT_OF[$m]} ${EXTRA_ARGS[$m]:-}"
    done
    echo ""
    echo "(dry run -- no jobs launched)"
    exit 0
fi

declare -a SLOT_PID SLOT_METHOD SLOT_LOG
for ((s = 0; s < NUM_GPUS; s++)); do
    SLOT_PID[s]=0
    SLOT_METHOD[s]=""
    SLOT_LOG[s]=""
done

_reap_slot() {
    local s="$1"
    wait "${SLOT_PID[$s]}"
    local code=$?
    if [[ $code -ne 0 ]]; then
        echo "[FAIL] gpu=${GPUS[$s]} method=${SLOT_METHOD[$s]} exit=$code  (log: ${SLOT_LOG[$s]})" | tee -a "$FAIL_LOG"
    else
        echo "[OK]   gpu=${GPUS[$s]} method=${SLOT_METHOD[$s]}"
    fi
    SLOT_PID[$s]=0
}

FREE_SLOT=-1
_find_free_slot() {
    # Runs inline in the main shell on purpose -- see header comment.
    while true; do
        for ((s = 0; s < NUM_GPUS; s++)); do
            if [[ "${SLOT_PID[$s]}" -eq 0 ]]; then
                FREE_SLOT=$s
                return
            fi
        done
        wait -n
        for ((s = 0; s < NUM_GPUS; s++)); do
            pid="${SLOT_PID[$s]}"
            if [[ "$pid" -ne 0 ]] && ! kill -0 "$pid" 2>/dev/null; then
                _reap_slot "$s"
            fi
        done
    done
}

idx=0
TOTAL=${#METHODS[@]}
while [[ $idx -lt $TOTAL ]]; do
    _find_free_slot
    slot=$FREE_SLOT
    method="${METHODS[$idx]}"
    gpu="${GPUS[$slot]}"
    log_file="$LOG_DIR/${method}.log"
    extra="${EXTRA_ARGS[$method]:-}"

    echo "[START $((idx + 1))/$TOTAL] gpu=$gpu method=$method  ${extra:+extra_args=\"$extra\"}"
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES="$gpu" python "${SCRIPT_OF[$method]}" $extra \
        > "$log_file" 2>&1 &

    SLOT_PID[$slot]=$!
    SLOT_METHOD[$slot]="$method"
    SLOT_LOG[$slot]="$log_file"
    idx=$((idx + 1))
done

for ((s = 0; s < NUM_GPUS; s++)); do
    if [[ "${SLOT_PID[$s]}" -ne 0 ]]; then
        _reap_slot "$s"
    fi
done

echo ""
if [[ -s "$FAIL_LOG" ]]; then
    echo "Done. Some methods FAILED -- see $FAIL_LOG:"
    cat "$FAIL_LOG"
else
    echo "Done. All $TOTAL method(s) completed successfully."
fi
echo ""
echo "Results: verify_results/<method>.csv"
