#!/usr/bin/env bash
# Same idea as scripts/organize_plots.sh, but for results/: moves every
# results/<name>/ directory that ISN'T new-pipeline output into
# results_legacy/, leaving only genuine scripts/run_pipeline.py output behind.
#
# "New-pipeline output" here means results/<name>/run_config.json exists --
# only written by results.write_run_config(), which only the new pipeline's
# `run` subcommand calls. None of the 57 legacy one-off scripts under
# scripts/legacy/ (or scripts/run_experiment.py's own results/ output) ever
# write it.
#
# Safe to run: the new pipeline never implicitly scans all of results/ --
# `run`/`merge`/`plot`/`detail-plot*` always take an explicit --run_name/
# --run/--runs, so moving legacy dirs elsewhere doesn't break anything the
# current pipeline does.
#
# Usage:
#   scripts/organize_results.sh            # move it
#   scripts/organize_results.sh --dry_run  # just print what would move, touch nothing

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
[[ "${1:-}" == "--dry_run" ]] && DRY_RUN=1

SRC="results"
DST="results_legacy"

if [[ ! -d "$SRC" ]]; then
    echo "No $SRC/ directory found, nothing to do."
    exit 0
fi

mkdir -p "$DST"

moved=0
kept=0

shopt -s nullglob
for entry in "$SRC"/*; do
    name="$(basename "$entry")"
    [[ "$name" == "$DST" ]] && continue   # safety: never touch a nested results_legacy

    if [[ -f "$entry/run_config.json" ]]; then
        echo "[keep] $entry  (has run_config.json)"
        kept=$((kept + 1))
    else
        echo "[move] $entry  -> $DST/$name"
        if [[ "$DRY_RUN" -eq 0 ]]; then
            mv "$entry" "$DST/$name"
        fi
        moved=$((moved + 1))
    fi
done

echo ""
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "(dry run) Would move $moved item(s), keep $kept item(s)."
else
    echo "Moved $moved item(s) to $DST/, kept $kept item(s) in $SRC/."
fi
