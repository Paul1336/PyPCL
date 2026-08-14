#!/usr/bin/env bash
# Step 5 experiment configs using each paper's ACTUAL original benchmark
# dataset (not the CIFAR-100 class-subset used by
# scripts/run_paper_alignment_experiments.sh). See docs/00_paper_alignment_guide.md's
# "資料集支援（--dataset）" section for the full dataset list and caveats.
#
# Every --dataset value below was verified end-to-end (real download + real
# training run, not just wired up) on 2026-08-14 -- see
# docs/dataset_availability_report.md and src/pipeline/datasets/*.py
# docstrings for exactly what was checked.
#
# Requires: pip install scipy ucimlrepo scikit-learn datasets  (for the
# UCI/text/CUB-200 loaders; MNIST-family and the real-world .mat datasets
# only need scipy). Also requires an `unrar` (or WinRAR) install on PATH for
# the 5 real-world PLL datasets and CLPL's own data (see
# docs/dataset_availability_report.md's manual follow-up notes).
set -euo pipefail

cd "$(dirname "$0")/.."

# ─── PRODEN / MCL-LOG / SCL-NL: MNIST-family (all three papers used it) ────
python scripts/run_pipeline.py run --run_name paper_original_mnist_family \
    --algorithms CLPL PRODEN MCL-LOG MCL-LOG-Fixed SCL-NL --dataset mnist --epochs 100
python scripts/run_pipeline.py run --run_name paper_original_mnist_family \
    --algorithms CLPL PRODEN MCL-LOG MCL-LOG-Fixed SCL-NL --dataset fashion-mnist --epochs 100
python scripts/run_pipeline.py run --run_name paper_original_mnist_family \
    --algorithms CLPL PRODEN MCL-LOG MCL-LOG-Fixed SCL-NL --dataset kmnist --epochs 100

# ─── PRODEN / MCL-LOG: UCI tabular real-world benchmarks ───────────────────
for ds in dermatology ecoli abalone yeast synthetic-control; do
    python scripts/run_pipeline.py run --run_name paper_original_uci \
        --algorithms CLPL PRODEN MCL-LOG MCL-LOG-Fixed --dataset "$ds" --epochs 200
done

# ─── MCL-LOG: 20 Newsgroups ─────────────────────────────────────────────────
python scripts/run_pipeline.py run --run_name paper_original_20news \
    --algorithms CLPL MCL-LOG MCL-LOG-Fixed --dataset 20newsgroups --epochs 100

# ─── PRODEN / MCL-LOG: the 5 classic real-world PLL benchmarks (real, not
# synthetic, candidate label sets). All 5 verified end-to-end 2026-08-14
# (download + extract + train), including Soccer Player / Yahoo!News. ──────
for ds in lost msrcv2 birdsong soccer-player yahoo-news; do
    python scripts/run_pipeline.py run --run_name paper_original_real_pll \
        --algorithms CLPL PRODEN MCL-LOG MCL-LOG-Fixed --dataset "$ds" --epochs 200
done

# ─── CLPL (Cour et al. 2011): its own original raw-image data ──────────────
python scripts/run_pipeline.py run --run_name paper_original_clpl_tv \
    --algorithms CLPL --dataset clpl-lost --epochs 100
python scripts/run_pipeline.py run --run_name paper_original_clpl_tv \
    --algorithms CLPL --dataset clpl-fiw --epochs 100

# ─── PiCO / ComCo: CUB-200 (real photos, HuggingFace mirror) ───────────────
python scripts/run_pipeline.py run --run_name paper_original_cub200 \
    --algorithms CLPL PRODEN PiCO PiCO-Fixed ComCo ComCo-Fixed --dataset cub200 --epochs 100

# ─── PiCO: CIFAR-100-H (hierarchical candidate labels, q=0.5) ──────────────
# Paper-exact generation confirmed via direct PDF text extraction (Section
# 4.4) -- see src/pipeline/datasets/cifar100_h.py's docstring. Use C close
# to 100 so coarse-superclass siblings are actually available (small
# C-subsets mostly fall back to uniform sampling, logged at runtime).
python scripts/run_pipeline.py run --run_name paper_original_cifar100h \
    --algorithms PiCO PiCO-Fixed --dataset cifar100-h --c_values 100 --epochs 200

# ─── ComCo: SUN397 (Section 5.6 supplementary experiment) ──────────────────
# UNVERIFIED (see src/pipeline/datasets/sun397.py docstring): code-complete,
# same lazy-path mechanism as cub200, but the ~17-37GB download was never
# triggered in the original pass. Run a small smoke test before trusting
# this, or before scaling up epochs/batch size.
# python scripts/run_pipeline.py run --run_name paper_original_sun397 \
#     --algorithms CLPL ComCo ComCo-Fixed --dataset sun397 --epochs 1 --batch_size 16

# ─── Merge everything into one results.csv per run_name ────────────────────
for run in paper_original_mnist_family paper_original_uci paper_original_20news \
           paper_original_real_pll paper_original_clpl_tv paper_original_cub200 \
           paper_original_cifar100h; do
    python scripts/run_pipeline.py merge --run "$run"
done
