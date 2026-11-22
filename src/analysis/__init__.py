"""Reusable analysis toolkit (everything beyond final test accuracy).

Training-time (pure functions on tensors; call inside an epoch loop):
    accum       EpochAccumulator (ratio-of-sums / mean-of-batch-means), CSV writers
    pairs       contrastive pair selection: precision / recall / violations / negative purity,
                within-batch confusion, PiCO-Oracle graded corrections
    labels      pseudo-label, confidence buffer, prototypes, candidate mass/entropy,
                EMA pseudo-labels, ComCo c*, clean-sample selection, candidate pruning, s_flip
    gradients   gradient alignment vs ground truth, bias/variance decomposition
    prediction  per-class loss/acc, entropy concentration, collapse, confusion, kNN, t-SNE
    estimates   label-free accuracy estimates (bandit rewards), inclusion matrix Pi

Post-hoc (pandas on saved artifacts):
    io          load_results / load_pcdiag / load_bandit_traces / load_prune_traces /
                load_alpha_traces / load_detail_csv
    stats       mean±std tables, two-level means, paired tests, oracle gains, best-per-column
    traces      epoch windows, final-epoch diagnostics, sign agreement / estimate bias,
                arm fractions / switches / settle epoch, prune + alpha summaries

Full inventory with the origin of each metric: docs/analysis_metrics.md.
Equivalence with the original implementations: verify_scripts/check_analysis.py.
"""

from . import accum, estimates, gradients, io, labels, pairs, prediction, stats, traces  # noqa: F401
from .accum import EpochAccumulator, RewritingCsvLog  # noqa: F401
