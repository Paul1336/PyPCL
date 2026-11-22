#!/usr/bin/env python
"""Single entry point for the PLL/CLL algorithm comparison pipeline.

Consolidates what used to be 57 scripts (run_sweep_*, run_*_comparison,
plot_*, launch_*.sh; scripts/legacy/ was removed 2026-11-22, see git history before 84d3cc9) into
one CLI: run / merge / report / plot / plot-weight / detail-plot* (14 subcommands; parsers in
src/pipeline/cli/parsers.py, implementations in src/pipeline/cli/commands.py). Sweeps of many cells:
scripts/sweep.py with experiments/<id>.yaml.

Subcommands:
    run              Train algorithms on the CIFAR-100 class-subset sweep and
                     record results (supports multi-GPU sharding and resume).
    merge            Merge per-worker result shards into
                     results/<run_name>/results.csv.
    plot             Draw accuracy-vs-k figures from one or more runs' merged
                     results.
    detail-plot      Per-class accuracy/loss heatmap over epoch checkpoints
                     (needs `run --detail`); one or two algorithms side by side.
    detail-plot-pico PiCO's contrastive pair-selection precision vs. ground
                     truth, over epochs (needs `run --detail`).

Examples:
    # Single GPU, a handful of algorithms
    python scripts/run_pipeline.py run --run_name demo \\
        --algorithms CLPL PRODEN MCL-LOG PiCO ComCo SoLar \\
        --c_values 5 20 --epochs 200

    # Multi-GPU: one process per GPU, same run_name, round-robin algorithm split
    CUDA_VISIBLE_DEVICES=0 python scripts/run_pipeline.py run --run_name demo --gpu_id 0 --num_gpus 8
    CUDA_VISIBLE_DEVICES=1 python scripts/run_pipeline.py run --run_name demo --gpu_id 1 --num_gpus 8
    ...

    # Merge / plot independently of training
    python scripts/run_pipeline.py merge --run demo
    python scripts/run_pipeline.py plot --runs demo --out plots/demo/summary.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.cli.commands import COMMANDS  # noqa: E402
from src.pipeline.cli.parsers import *  # noqa: E402,F401,F403
from src.pipeline.cli.parsers import (_add_run_parser, _add_merge_parser, _add_report_parser, _add_plot_parser, _add_plot_weight_parser, _add_detail_plot_parser, _add_detail_plot_multi_parser, _add_detail_plot_pico_parser, _add_detail_plot_pico_confusion_parser, _add_detail_plot_pico_confusion_all_parser, _add_detail_plot_pico_multik_parser, _add_detail_plot_concentration_parser, _add_detail_plot_knn_parser, _add_detail_plot_pico_oracle_parser)  # noqa: E402,F401


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    _add_run_parser(sub)
    _add_merge_parser(sub)
    _add_report_parser(sub)
    _add_plot_parser(sub)
    _add_plot_weight_parser(sub)
    _add_detail_plot_parser(sub)
    _add_detail_plot_multi_parser(sub)
    _add_detail_plot_pico_parser(sub)
    _add_detail_plot_pico_confusion_parser(sub)
    _add_detail_plot_pico_confusion_all_parser(sub)
    _add_detail_plot_pico_multik_parser(sub)
    _add_detail_plot_pico_oracle_parser(sub)
    _add_detail_plot_concentration_parser(sub)
    _add_detail_plot_knn_parser(sub)
    args = parser.parse_args()

    return COMMANDS[args.command](args)


if __name__ == '__main__':
    main()
