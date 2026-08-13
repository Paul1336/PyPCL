# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PyPCL** is a Python framework for comparing **Partial Label Learning (PLL)** and **Complementary Label Learning (CLL)** algorithms on CIFAR datasets. The `src/pipeline` algorithm registry (see below) covers 14 algorithms across both paradigms:
- **PLL algorithms**: CLPL (Cour 2011), Wu2022, PRODEN, PiCO, PiCO-MCL, PiCO-SC, PiCO-CLS, SoLar
- **CLL algorithms**: MCL-LOG, SCL-NL, OP, OP-W, CPE, ComCo

There are two independent workflows in this repo:
1. **Algorithm comparison pipeline** (`scripts/run_pipeline.py`) — the actively used workflow. Trains the algorithms above on CIFAR-100 class-subsets (configurable number of classes `C` and candidate-label count `k`) to compare accuracy across methods. See "Running the Comparison Pipeline" below.
2. **Full-dataset experiment runner** (`scripts/run_experiment.py`) — trains PiCO, Proden, SoLar, and MCL (LOG/MAE/EXP) on complete CIFAR-10/CIFAR-20/CLCIFAR-10/CLCIFAR-20 datasets. See "Running the Full-Dataset Experiment" below.

## Installation & Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Key dependencies: torch, torchvision (CUDA 11.7), PyYAML, numpy, PIL, matplotlib, pandas, tqdm
```

## Running the Comparison Pipeline

`scripts/run_pipeline.py` is the single entry point for comparing all 14 algorithms on CIFAR-100 class-subsets. It replaced 57 one-off scripts (`run_sweep_*`, `run_*_comparison`, `plot_*`, `launch_*.sh`) that had accumulated ad hoc, largely duplicated CLI parsing / CSV handling / plotting logic — those are kept for reference under `scripts/legacy/` but should not be used for new work.

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

## Running the Full-Dataset Experiment

`scripts/run_experiment.py` is a separate workflow: it trains PiCO, Proden, SoLar, and MCL (LOG/MAE/EXP) on complete CIFAR-10/20/CLCIFAR-10/20 datasets (not the CIFAR-100 class-subset sweep above). Hyperparameters can be set via command-line arguments (which override `config.yaml` defaults).

### Basic Examples

```bash
# CIFAR-10: constant setting with k=2 partial labels, clean labels
python scripts/run_experiment.py --dataset cifar10 --type constant --value 2 --noise clean

# CIFAR-20: variable setting with 50% label inclusion probability, 20% label noise
python scripts/run_experiment.py --dataset cifar20 --type variable --value 0.5 --noise noisy --eta 0.2

# CLCIFAR-10: human-annotated dataset (no type/value/noise required)
python scripts/run_experiment.py --dataset clcifar10
```

### Command-Line Arguments

| Argument | Values | Purpose |
|----------|--------|---------|
| `--dataset` | `cifar10`, `cifar20`, `clcifar10`, `clcifar20` | Dataset selection |
| `--type` | `constant`, `variable` | Label generation strategy (CIFAR-10/20 only) |
| `--value` | float | k for constant, q for variable |
| `--noise` | `clean`, `noisy` | Add label noise (CIFAR-10/20 only) |
| `--eta` | float | Noise level (0.0-1.0) |
| `--batch_size` | int | Training batch size |
| `--epochs` | int | Total training epochs |
| `--lr` | float | Learning rate |
| `--weight_decay` | float | SGD weight decay |
| `--momentum` | float | SGD momentum |

Results (accuracies, plots) are saved to `results/` and `plots/` directories.

## Architecture & Data Flow

### High-Level Pipeline

1. **Data Preparation** (`src/data_setup.py`)
   - Load base CIFAR/CLCIFAR dataset
   - Generate weak labels using `ComparisonDataGenerator` (for CIFAR) or use pre-annotated labels (for CLCIFAR)
   - Create separate dataloaders for each algorithm's needs

2. **Algorithm Training** (`src/training_pipelines.py` → `src/engine.py`)
   - Each algorithm has a dedicated setup function in `src/model_setup.py`
   - Algorithms are trained sequentially, with memory cleanup between runs
   - Per-epoch accuracies are tracked and plotted

3. **Results & Visualization** (`src/saving.py`, `src/plotting.py`)
   - Accuracies saved to CSV with hyperparameter metadata
   - Accuracy plots generated and saved after each algorithm completes

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
├── args.py                    # Argument parsing & validation
├── data_setup.py              # Dataset preparation, loader creation
├── data_utils.py              # Dataset classes (Weakly Supervised, PiCO, SoLar)
├── engine.py                  # Core training loops: train_algorithm, train_pico_epoch, train_solar_epoch, train_solar
├── models.py                  # ResNet18 backbone
├── model_setup.py             # Algorithm-specific initialization (setup_proden, setup_mcl, setup_pico, setup_solar)
├── training_pipelines.py      # Entry points for each algorithm (run_*_training)
├── collate.py                 # Dataloader collate functions
├── proden_loss.py             # Proden loss implementation
├── mcl_losses.py              # MCL loss implementations (LOG, MAE, EXP)
├── saving.py                  # Results to CSV
├── plotting.py                # Accuracy plot generation
├── clcifar.py                 # CLCIFAR10/20 dataset loaders (downloads from Google Drive)
├── pico/
│   ├── model.py               # PiCO model (dual encoder, queue, prototypes)
│   ├── resnet.py              # SupConResNet backbone with feature head
│   ├── utils_loss.py          # PartialLoss, SupConLoss (contrastive)
│   └── randaugment.py         # RandAugment augmentation implementation
├── solar/
│   ├── utils_algo.py          # Sinkhorn-Knopp algorithm, linear rampup
│   └── utils_loss.py          # Partial loss with confidence tracking
└── pipeline/                  # Comparison pipeline (scripts/run_pipeline.py) — see above
    ├── config.py              # Resolved run config: merges config.yaml + CLI args
    ├── gpu.py                 # Device selection, multi-GPU round-robin algorithm sharding
    ├── data.py                # C/k schedule + CIFAR-100 subset dataloaders (wraps src/cifar100_subset.py)
    ├── results.py             # Unified results CSV schema, shard merge/resume
    ├── plotting.py            # Single parameterized accuracy-vs-k plotting tool
    ├── runner.py               # Experiment loop: for C, k, algorithm -> train -> record
    └── algorithms/
        ├── __init__.py        # Algorithm registry: name -> (paradigm, runner)
        ├── hparams.py         # Per-algorithm default hyperparameters
        └── runners.py         # Training "shapes" (simple / proden / pico family / comco / solar)
scripts/
├── run_pipeline.py            # Main entry point: 14-algorithm CIFAR-100 subset comparison pipeline
├── run_experiment.py          # Full-dataset entry point: CIFAR-10/20/CLCIFAR-10/20, 7 algorithms
├── download_data.py           # One-time CIFAR-10/100 download utility
└── legacy/                    # Superseded one-off scripts, kept for reference only
config.yaml                    # Hyperparameter defaults
```

## Key Design Patterns

### 1. Modular Algorithm Design
- Each algorithm has three components:
  - **Setup** (`setup_*` in model_setup.py): Initialize model, loss, optimizer
  - **Training** (`train_*` in training_pipelines.py or engine.py): Run training loop
  - **Dataloader** (get_dataloaders): Provide algorithm-specific loader
- Enables easy addition of new algorithms

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
