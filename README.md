# Python Partial and Complementary Label Learning Library

This project compares complementary label learning (CLL) and partial label learning (PLL). It provides a Python framework for experimenting with state-of-the-art algorithms on CIFAR datasets.

## Implemented Strategies

The library implements 14 algorithms for PLL and CLL (see `src/pipeline/algorithms/__init__.py`
for the full registry). The table below covers the 6 that have had a paper-vs-code fidelity pass
completed (2026-08-14) against the original paper PDFs — see the linked doc for each one's
equation-by-equation comparison, confirmed benchmark, and (where a bug was found) the corrected
`*-Fixed` algorithm ID. The other 8 (`Wu2022`, `SoLar`, `PiCO-MCL`, `PiCO-SC`, `PiCO-CLS`, `OP`,
`OP-W`, `CPE`) are implemented and runnable but have not yet had this fidelity pass.

| Strategy | Type | Description | Fidelity doc | Verified against paper? |
| --- | --- | --- | --- | --- |
| [PiCO](https://arxiv.org/abs/2201.08984) | PLL | Contrastive label disambiguation via dual-encoder + class prototypes. | [pico_explanation.md](docs/pico_explanation.md) | ✅ core loss/model exact match; ⚠️ warm-up schedule differs — use `PiCO-Fixed` |
| [PRODEN](https://arxiv.org/abs/2002.08053) | PLL | Progressive identification: EM-style candidate-label reweighting each mini-batch. | [proden_explanation.md](docs/proden_explanation.md) | ✅ exact match (`ProdenLoss`, used by the main pipeline) |
| [CLPL](https://jmlr.org/papers/v12/cour11a.html) | PLL | Cour, Sapp & Taskar (JMLR 2011): squared-hinge margin loss on candidate label sets. | [cour2011_explanation.md](docs/cour2011_explanation.md) | ✅ exact match |
| [MCL-LOG](https://proceedings.mlr.press/v119/feng20a.html) | CLL | Unbiased-risk-estimator loss over multiple complementary labels (log variant). | [mcl_explanation.md](docs/mcl_explanation.md) | ⚠️ URE scaling factor was wrong — use `MCL-LOG-Fixed` |
| [SCL-NL](https://proceedings.mlr.press/v119/chou20a.html) | CLL | Surrogate (non-URE) "negative learning" complementary loss. | [scl_nl_explanation.md](docs/scl_nl_explanation.md) | ✅ single-CL formula exact match; multi-CL averaging is an unverified repo extension |
| [ComCo](https://www.sciencedirect.com/science/article/pii/S0893608023005683) | CLL | Complementary supervised contrastive learning (MoCo-style dual encoder + queue). | [comco_explanation.md](docs/comco_explanation.md) | ✅ contrastive loss exact match; ⚠️ classification-loss scaling was wrong — use `ComCo-Fixed` |

See [docs/00_paper_alignment_guide.md](docs/00_paper_alignment_guide.md) for the full paper↔code
mapping, known-issues list, and the methodology behind the fidelity checks above.

## Dataset Generation

The script generates PLL and CLL datasets from **CIFAR-10** and **CIFAR-20**. It also supports the human-annotated **CLCIFAR-10** and **CLCIFAR-20** datasets.

For generated datasets, the following parameters are available:

| Parameter | Options | Description |
| --- | --- | --- |
| `Type` | `constant`, `variable` | Specifies if the number of labels per sample is fixed. |
| `Value` | | - If `constant`, `Value` (k) is the number of partial labels. <br> - If `variable`, `Value` (q) is the probability of a false label being included. |
| `Noise` | `noisy`, `clean` | Introduces label noise if set to 'noisy'. |
| `eta` | | Sets the noise level for noisy data. |

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/esoau/pypcl.git](https://github.com/esoau/pypcl.git)
    cd pypcl
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Comparing algorithms (recommended)

`scripts/run_pipeline.py` trains and compares all algorithms above on CIFAR-100 class-subsets (choose how many classes `C` and how many candidate labels `k` per sample):

```bash
python scripts/run_pipeline.py run --run_name demo \
    --algorithms CLPL PRODEN MCL-LOG PiCO ComCo SoLar \
    --c_values 5 20 --epochs 200

python scripts/run_pipeline.py plot --runs demo --out plots/demo/summary.png
```

It also supports multi-GPU sharding (`--gpu_id`/`--num_gpus`), resuming interrupted runs, and merging results independently of training (`merge` subcommand). See `CLAUDE.md` for the full option list and `src/pipeline/` for the implementation.

#### Training on a paper's actual original benchmark (`--dataset`)

By default the pipeline trains on CIFAR-100 class-subsets, which is *not* what most of the 6
papers above originally benchmarked on (see each `docs/*_explanation.md`'s "原論文使用的
Benchmark" section). Pass `--dataset` to train on the paper's real benchmark instead — every value
below was verified end-to-end (real download + real training run), not just wired up:

```bash
# PRODEN / MCL-LOG / SCL-NL's own benchmarks (MNIST family, UCI tabular, 20 Newsgroups)
python scripts/run_pipeline.py run --run_name mnist_demo --dataset mnist \
    --algorithms CLPL PRODEN MCL-LOG SCL-NL --epochs 100

python scripts/run_pipeline.py run --run_name dermatology_demo --dataset dermatology \
    --algorithms CLPL PRODEN MCL-LOG --epochs 200

# The 5 classic real-world PLL benchmarks (REAL candidate label sets, not synthetic)
python scripts/run_pipeline.py run --run_name lost_demo --dataset lost \
    --algorithms CLPL PRODEN --epochs 200

# CLPL's (Cour et al. 2011) own original raw-image data
python scripts/run_pipeline.py run --run_name clpl_lost_demo --dataset clpl-lost \
    --algorithms CLPL --epochs 100

# PiCO/ComCo's CUB-200 and CIFAR-100-H settings
python scripts/run_pipeline.py run --run_name cub200_demo --dataset cub200 \
    --algorithms CLPL PRODEN PiCO --epochs 100
```

Full dataset list, per-dataset caveats (some skip PiCO/ComCo/SoLar — no image augmentation story
for grayscale/tabular data), and verification notes: see
[docs/00_paper_alignment_guide.md](docs/00_paper_alignment_guide.md)'s "資料集支援" section and
[docs/dataset_availability_report.md](docs/dataset_availability_report.md).

### Full-dataset experiments

`run_experiment.py` trains PiCO, Proden, SoLar, and MCL on complete CIFAR-10/CIFAR-20/CLCIFAR datasets. Hyperparameters can be set via command-line arguments or in `config.yaml`.

| Parameter | Description |
| --- | --- |
| `--dataset` | Dataset to use (`cifar10`, `cifar20`, `clcifar10`, `clcifar20`). |
| `--batch_size` | Batch size for training. |
| `--epochs` | Number of training epochs. |
| `--lr` | Learning rate. |
| `--weight_decay` | Weight decay for the optimizer. |
| `--momentum` | Momentum for the optimizer. |

**Example 1:**

Train models with a constant of 2 partial labels and no noise:

```bash
python scripts/run_experiment.py --dataset cifar10 --type constant --value 2 --noise clean
```

**Example 2:**

CIFAR-20 with Variable Labels and Noise Run all algorithms on CIFAR-20. First, 20% label noise is applied to the ground truth. Then, partial labels are generated where each false label has a 50% probability of being included in the candidate set.

```bash
python scripts/run_experiment.py --dataset cifar20 --type variable --value 0.5 --noise noisy --eta 0.2
```

**Example 3:**

CLCIFAR-10 (Human-Annotated) Run all algorithms on the pre-defined CLCIFAR-10 dataset.

```bash
python scripts/run_experiment.py --dataset clcifar10
```
