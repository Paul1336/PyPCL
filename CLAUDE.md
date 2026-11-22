# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PyPCL** is a Python framework for comparing **Partial Label Learning (PLL)** and **Complementary Label Learning (CLL)** algorithms on CIFAR datasets. The `src/pipeline` algorithm registry (see below) covers 14 algorithms across both paradigms:
- **PLL algorithms**: CLPL (Cour 2011), Wu2022, PRODEN, PiCO, PiCO-MCL, PiCO-SC, PiCO-CLS, SoLar
- **CLL algorithms**: MCL-LOG, SCL-NL, OP, OP-W, CPE, ComCo

Workflow:
1. **Algorithm comparison pipeline** (`scripts/run_pipeline.py`) — one training cell or a small sweep. Trains the algorithms above on CIFAR-100 class-subsets (configurable number of classes `C` and candidate-label count `k`) to compare accuracy across methods. See "Running the Comparison Pipeline" below.
2. **Declarative sweeps** (`scripts/sweep.py experiments/<id>.yaml`) — the launcher for multi-cell experiments (GPU packing, resume, retries); see `experiments/README.md`.

The old full-dataset workflow (`scripts/run_experiment.py`, `src/{args,data_setup,training_pipelines,saving,plotting,clcifar,model_setup}.py`) and `scripts/legacy/` were removed in the 2026-11-22 refactor (`docs/refactor_plan.md`); they are in git history before 84d3cc9.

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Key dependencies: torch, torchvision (CUDA 11.7), PyYAML, numpy, PIL, matplotlib, pandas, tqdm
```

## Running the Comparison Pipeline

`scripts/run_pipeline.py` is the single entry point for comparing all 14 algorithms on CIFAR-100 class-subsets. It replaced 57 one-off scripts (`run_sweep_*`, `run_*_comparison`, `plot_*`, `launch_*.sh`) that had accumulated ad hoc, largely duplicated CLI parsing / CSV handling / plotting logic (removed 2026-11-22).

It has three subcommands:

```bash
# Train: for each C in --c_values, sweeps a k schedule (candidate-label counts) and
# trains every algorithm in --algorithms, writing results as it goes (resumable).
python scripts/run_pipeline.py run --run_name demo \
    --algorithms CLPL PRODEN MCL-LOG PiCO ComCo SoLar \
    --c_values 5 20 --epochs 200

# Multi-GPU: one process per GPU, same --run_name, algorithms are round-robin split
# across --num_gpus workers (or pin one worker to one algorithm with --algo).
CUDA_VISIBLE_DEVICES=0 python scripts/run_pipeline.py run --run_name demo --gpu_id 0 --num_gpus 8
CUDA_VISIBLE_DEVICES=1 python scripts/run_pipeline.py run --run_name demo --gpu_id 1 --num_gpus 8
# ...

# Merge per-GPU result shards into results/<run_name>/results.csv (also runs
# automatically after every training cell during `run`).
python scripts/run_pipeline.py merge --run demo

# Draw accuracy-vs-k figures from one or more runs' merged results.
python scripts/run_pipeline.py plot --runs demo --out plots/demo/summary.png
```

Results land in `results/<run_name>/` (`run_config.json` snapshot, per-worker `shards/worker<id>.csv`, merged `results.csv`); plots default to `plots/<run_name>/`. See `src/pipeline/` for the implementation (`gpu.py` device/sharding, `data.py` C/k schedule + dataloaders, `algorithms/` registry + per-algorithm hyperparameters + training runners, `results.py` CSV schema, `plotting.py`, `runner.py` ties it together). Adding a new algorithm only requires a runner function in `src/pipeline/algorithms/runners.py`, a hyperparameter entry in `hparams.py`, and one line in the registry (`src/pipeline/algorithms/__init__.py`).

## Architecture & Data Flow

### High-Level Pipeline

`scripts/run_pipeline.py run` -> `src/pipeline/runner.py` (C x seed x k x algorithm loop, one `CellContext` per cell, resume by shard + `config_hash`) -> `src/pipeline/data.py` (`src/cifar100_subset.py` or the `src/pipeline/datasets` registry) -> registry `src/pipeline/algorithms` (507 names) -> runner:
- PiCO / ComCo variants: `algorithms/families.py` (`run_pico_family`, `run_comco_family`, shared `run_epochs` with the diagnostic hooks) -> one epoch function each in `src/train/pico.py` / `src/train/comco.py` (variants = `PiCOSpec` / `ComCoSpec` flags; oracle corrections = mask transforms; detail stats = `src/train/hooks.py`).
- everything else: `algorithms/runners/{simple,sweeps,bandit,swap}.py`; diagnostic-heavy re-implementations in `src/pcdiag.py` (PCD-*) and `src/picoplus.py` (PCD-PiCOPlus*).
- losses: `src/losses/` (canonical formulas in `functional.py`); models: `src/models/`, `src/pico/model.py`, `src/comco/model.py`.
- results: `results/<run>/shards/worker<id>.csv` (+ `results.csv`, `run_configs/<config_hash>.json`), diagnostics under `results/<run>/{pcdiag,detail,bandit_trace,prune_trace,alpha_trace}/`; analysis with `src/analysis/`.

### Label Generation Strategies

**Partial Label Learning (PL)**: Each sample has a set of candidate labels containing the true label plus false candidates.
- **Constant**: Fixed k candidate labels per sample (k must be > 1 and ≤ num_classes)
- **Variable**: Each false label independently included with probability q

**Complementary Label Learning (CL)**: Each sample has a set of labels known to NOT contain the true label.
- Derived as the complement of PL labels (or provided directly in CLCIFAR datasets)

Noise can be applied: with probability η, the true label is removed from PL sets (making them potentially empty).

### Dataset Classes

| Class | Purpose |
|-------|---------|
| `WeaklySupervisedDataset` | Standard wrapper for images + weak labels, applies optional transforms |
| `PicoDataset` | Provides weak/strong augmented image pairs + one-hot PL vectors for PiCO |
| `SoLarDataset` | Provides weak/strong augmented pairs + one-hot PL vectors for SoLar |

Collate functions (`src/collate.py`) handle variable-length label tensors by padding with -1.

### Algorithm-Specific Details

#### 1. Proden (src/proden_loss.py)
- **Loss**: Weighted average of log-likelihoods over partial labels
- Weights are normalized softmax probabilities of model predictions on candidate labels
- Simple, baseline approach for PL learning
- Dataloader: Standard `pl_loader`

#### 2. MCL (src/mcl_losses.py)
- **Three loss variants**:
  - **MCL-LOG**: `-log(Σ P(non-complementary labels))`
  - **MCL-MAE**: `1 - Σ P(non-complementary labels)`
  - **MCL-EXP**: `exp(-Σ P(non-complementary labels))`
- Unbiased risk estimator scaling: `(C-1)/(C-m)` where m = number of complementary labels
- Dataloader: Standard `cl_loader`

#### 3. PiCO (src/pico/)
- **Architecture**: Dual-encoder (momentum-updated key encoder) with momentum contrast (MoCo) queue
- **Key components**:
  - `SupConResNet`: ResNet18 with supervised contrastive feature head (low_dim=128)
  - Prototype memory per class, updated via exponential moving average
  - Feature queue for contrastive loss
- **Training**:
  - Partial loss: `-Σ(confidence[i] * log_softmax(cls_out)[i])`
  - Confidence updated via EMA based on prototype scores (after warmup epoch)
  - Supervised contrastive loss over weak/strong augmented pairs + queue
  - Combined loss: `cls_loss + loss_weight * cont_loss`
- **Dataloader**: Special `pico_loader` with custom augmentations (RandomAugment)
- **Config parameters**: low_dim, moco_queue, moco_m (0.999), proto_m (0.99), prot_start (warmup epoch), conf_ema_range, loss_weight

#### 4. SoLar (src/solar/)
- **Two-stage training**:
  - **Stage 1 (Pre-estimation, 100 epochs)**: Estimate empirical class distribution from model predictions
  - **Stage 2 (Final Training, variable epochs)**: Use refined distribution + Sinkhorn-Knopp algorithm
- **Key algorithm**: Sinkhorn-Knopp (`src/solar/utils_algo.py`) normalizes cost matrix to match target marginal distribution
  - Solves optimal transport problem for label disambiguation
  - Cost = model softmax predictions * partial label indicator
- **Training loop**:
  - Sinkhorn selects pseudo-labels with confidence threshold
  - Hard/soft label selection based on loss and confidence
  - Mixup augmentation for selected samples
  - Loss combines pseudo-label loss, consistency loss, and unreliable sample loss
  - Empirical distribution updated via EMA (gamma1=0.1 stage 1, gamma2=0.01 stage 2)
- **Queue mechanism**: Maintains historical predictions for better distribution estimation
- **Dataloader**: Special `solar_loader` with weak/strong augmentations
- **Config parameters**: warmup_epoch, rho_range, lamd (lambda), eta, tau (confidence threshold), est_epochs, gamma1/gamma2 (EMA rates)

## File Organization

```
src/
├── core/            CellContext (per-cell identity), trace writer
├── data/            transforms.py: normalisation constants + PiCO/ComCo weak/strong augmentation
├── losses/          functional (canonical formulas), masks, stateless / confidence / composite / swap / contrastive
├── models/          backbones (ResNet18 / MLP), common (key-encoder momentum, prototype EMA, predict_logits)
├── train/           pico.py / comco.py (the one epoch function per framework + specs), hooks.py
├── pipeline/
│   ├── runner.py  config.py  data.py  results.py  gpu.py  plotting.py  detail.py  biasvariance*.py
│   ├── cli/         run_pipeline parsers + subcommand implementations
│   ├── datasets/    non-CIFAR-100 dataset registry
│   └── algorithms/  __init__ (registry, needs_views), hparams (rule), families.py, runners/{simple,sweeps,bandit,swap}.py
├── sweep/           spec.py (YAML -> cells -> argv), scheduler.py (GPU packing, resume, retries)
├── analysis/        non-accuracy diagnostics (training-time + post-hoc); inventory docs/analysis_metrics.md
├── pico/ comco/ solar/   framework-specific models / contrastive loss / SoLar utilities
├── cifar100_subset.py  data_utils.py  collate.py  pll_init.py   data generation, datasets, confidence inits
├── pcdiag.py  picoplus.py  clsloss_swap.py  clsloss_solo.py  comco_bandit.py  bias_aware_reward.py
├── hybrid_switch.py  solo_switch.py  switch_diag.py  unified_loss.py
└── *_loss.py, engine.py, fixed_*_engine.py, oracle_pico_engine.py   thin re-export shims kept for old imports
scripts/   run_pipeline.py, sweep.py, download_data.py, log_experiment.py, plot_*.py, summarize_clsloss_swap.py,
           launch_clsloss_swap.py (build_plan = Nano4 shard map), run_clsloss_solo.py, run_biasvariance_faithful.py
experiments/  one YAML spec per sweep
tests/     run_all.py (safety net), golden/, step_equiv.py, unit_checks.py, sweep_equiv.py, legacy_ref/ (frozen pre-refactor src)
config.yaml                    Hyperparameter defaults
```

## Before changing training code

Run `python tests/run_all.py` (or `--fast` without the ~10 min GPU golden cells). Golden cells
(`tests/golden/`) fingerprint 97 representative algorithms deterministically; a deliberate behaviour
change is accepted with `golden.py check --allow <regex> --update` after checking that only the
expected cells/columns changed. Do not edit `tests/legacy_ref/`.

## Key Design Patterns

### 1. Modular Algorithm Design
- A new PiCO/ComCo variant is usually a new `PiCOSpec`/`ComCoSpec` flag or `PiCOVariant` entry plus one registry name; other algorithms get a runner in `algorithms/runners/` and a registry line. Hyperparameters follow `hparams.hparams_for`.

### 2. Configuration Management
- `config.yaml` provides defaults for all hyperparameters
- Command-line arguments override config values
- Algorithm-specific configs (pico, solar) passed to setup/training functions

### 3. Memory Efficiency
- Models deleted and garbage collected after training
- GPU cache cleared periodically during long runs
- Queue/buffer mechanisms (PiCO, SoLar) manage historical data efficiently

### 4. Weak Label Handling
- Flexible representation: PyTorch tensors (variable-length lists converted to padded tensors)
- Padding with -1 indicates invalid labels (masked in loss computations)
- One-hot encoding for algorithms requiring dense label matrices

### 5. Augmentation Strategy
- **Standard transforms**: Random crop, horizontal flip, color jitter (CIFAR normalization)
- **Weak augmentation**: Minimal transforms for consistency
- **Strong augmentation**: RandAugment (n=3, m=5) for contrastive/semi-supervised methods

## Config.yaml Reference

Key sections:
- `data_generation`: CIFAR path, default noise level (eta)
- `training`: Batch size, epochs, learning rate, momentum, weight decay, num_classes
- `pico`: Feature dimension, MoCo queue size, EMA coefficients, warmup schedules
- `solar`: Warmup epochs, selection thresholds, Sinkhorn lambda, EMA rates

## Device & Performance Notes

- CUDA detection: `torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Torch version pinned to 1.13.1+cu117 (V100 compatibility)
- Batch size = 512 by default (adjust for GPU memory constraints)
- Full run (all 6 algorithms, 1000 epochs): ~8-12 hours on single GPU
