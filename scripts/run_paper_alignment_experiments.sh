#!/usr/bin/env bash
# Step 5 experiment configs for the 6 papers covered by the 2026-08-14 paper-alignment
# pass (see docs/00_paper_alignment_guide.md). Requires the new pipeline
# (scripts/run_pipeline.py) to be verified working (real CIFAR-100, one successful
# smoke-test run) before running any of this for real.
#
# Each block is independent and resumable (run_pipeline.py's `run` subcommand skips
# already-completed (C, k, algorithm) cells), so blocks can be run in any order, split
# across GPUs (--gpu_id/--num_gpus), or re-run after interruption.
set -euo pipefail

cd "$(dirname "$0")/.."

# ─── Group A: bugfix verification ──────────────────────────────────────────
# Runs each buggy algorithm next to its -Fixed counterpart across the default
# C x k sweep. For MCL-LOG/ComCo the bug only shows up when m (= C - k, the
# complementary-label count) is large, i.e. small k -- watch the low-k end of
# the accuracy-vs-k curve. For PiCO the difference should show up mainly in
# early-training loss curves (per-epoch, not just final accuracy) since the
# discrepancy is in the warm-up schedule.
python scripts/run_pipeline.py run --run_name paper_alignment_bugfix_check \
    --algorithms MCL-LOG MCL-LOG-Fixed ComCo ComCo-Fixed PiCO PiCO-Fixed \
    --c_values 5 20 --epochs 200

# ─── Group B: confirmed-faithful baselines ─────────────────────────────────
# CLPL, PRODEN (ProdenLoss via run_proden), and SCL-NL were all verified to
# match their papers exactly (no fixed_ version produced) -- run as-is,
# mainly to have them in the same results.csv for cross-algorithm plots.
python scripts/run_pipeline.py run --run_name paper_alignment_bugfix_check \
    --algorithms CLPL PRODEN SCL-NL \
    --c_values 5 20 --epochs 200

# ─── Merge + plot the combined run ─────────────────────────────────────────
python scripts/run_pipeline.py merge --run paper_alignment_bugfix_check
python scripts/run_pipeline.py plot --runs paper_alignment_bugfix_check \
    --out plots/paper_alignment_bugfix_check/summary.png

# ─── Group C (optional): paper-faithful hyperparameters ───────────────────
# Every paper's own training regime (batch size, epoch count, sometimes
# optimizer) differs from this repo's shared cross-algorithm defaults -- see
# each doc's "原論文使用的 Benchmark" section for the exact reported values.
# This is NOT a reproduction of the papers' results (different dataset:
# CIFAR-100 class-subsets here vs. each paper's own benchmark -- CLPL in
# particular used no CIFAR data at all, see docs/cour2011_explanation.md) --
# it only brings the *training regime* closer, as an additional data point.
#
# MCL-LOG / MCL-LOG-Fixed: paper used Adam (already the default here),
# batch 256, 250 epochs, ResNet-34/DenseNet-22 (this repo: ResNet-18 always).
python scripts/run_pipeline.py run --run_name paper_alignment_paperlike_mcl \
    --algorithms MCL-LOG MCL-LOG-Fixed \
    --c_values 5 20 --epochs 250 --batch_size 256

# PRODEN: paper used SGD momentum 0.9 (already the default here), batch 256,
# 500 epochs, 32-layer ResNet (this repo: ResNet-18).
python scripts/run_pipeline.py run --run_name paper_alignment_paperlike_proden \
    --algorithms PRODEN \
    --c_values 5 20 --epochs 500 --batch_size 256

# PiCO-Fixed with the paper's harder-setting warm-up (100 epochs, only used
# by the paper for CIFAR-100 @ q=0.1): requires overriding config.yaml's
# `pico.prot_start_fixed` to 100 before running (see docs/pico_explanation.md).
# python scripts/run_pipeline.py run --run_name paper_alignment_paperlike_pico \
#     --algorithms PiCO-Fixed --c_values 20 --epochs 800 --batch_size 256
