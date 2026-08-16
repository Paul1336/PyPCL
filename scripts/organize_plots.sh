#!/usr/bin/env bash
# One-time cleanup: moves everything currently under plots/ that was NOT
# produced by the new pipeline (scripts/run_pipeline.py's plot / detail-plot
# / detail-plot-pico subcommands) into plots_legacy/, leaving only genuine
# new-pipeline output behind.
#
# "New-pipeline output" is determined by: does results/<name>/run_config.json
# exist for that plots/<name> entry? run_config.json is only ever written by
# results.write_run_config(), which only the new pipeline's `run` subcommand
# calls -- none of the 57 legacy one-off scripts under scripts/legacy/ write
# it. A plots/<name> entry with no matching results/<name>/run_config.json is
# either legacy-script output or an orphaned one-off plot with no matching
# run_name at all -- both get moved, since neither could have come from the
# current pipeline.
#
# Usage:
#   scripts/organize_plots.sh            # move it
#   scripts/organize_plots.sh --dry_run  # just print what would move, touch nothing

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=0
[[ "${1:-}" == "--dry_run" ]] && DRY_RUN=1

SRC="plots"
DST="plots_legacy"

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
    [[ "$name" == "$DST" ]] && continue   # safety: never touch a nested plots_legacy

    # Strip a trailing file extension for loose files (e.g. foo.png -> foo)
    # so a plot named after its run_name (e.g. plots/demo.png) can still
    # match results/demo/run_config.json.
    base="$name"
    if [[ -f "$entry" && "$name" == *.* ]]; then
        base="${name%.*}"
    fi

    if [[ -f "results/$base/run_config.json" ]]; then
        echo "[keep] $entry  (matches results/$base/run_config.json)"
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
