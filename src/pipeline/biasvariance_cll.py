"""Gradient bias-variance diagnostic for the `scripts/run_biasvariance_faithful.py`
experiment -- reproduces the protocol of Chou et al. ("Unbiased Risk
Estimators Can Mislead: A Case Study of Learning with Complementary Labels",
ICML 2020) Section 4 as closely as this repo's data/model stack allows, in
contrast to src/pipeline/biasvariance.py (the PiCO-family diagnostic, which
deliberately diverges from the paper's protocol -- see that module's
docstring and docs/scl_nl_explanation.md).

Two protocol differences from src/pipeline/biasvariance.py, both taken
directly from the paper (docs/scl_nl_explanation.md, "訓練設定" /
"梯度分析(Section 4)"):

  1. The reference model is trained ONLY on the true label (plain CE, fixed
     SGD lr, no momentum/schedule) -- never on any complementary-label loss.
     Every complementary loss under test is evaluated OFF this frozen
     trajectory, purely as a diagnostic; it never contributes to a parameter
     update. This isolates each loss's gradient-estimation quality from
     confounds of "what a model looks like after actually being trained
     with it" (PiCO-family's PartialLoss confidence buffer, etc.).

  2. Complementary labels are single-CL (one uniformly-random wrong class
     per sample, Sec 2.2 of the paper), not k-sized candidate sets -- the
     paper's theory and Section 4 experiments only ever consider |Ybar|=1.
"""

import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.pipeline.biasvariance import oracle_cls_grad, _reduce

FIELDS = ['dataset', 'C', 'algorithm', 'seed', 'epoch', 'bias2', 'variance', 'mse', 'cos_sim']


def resample_single_cl_batch(true_labels: torch.Tensor, C: int, seed_val: int, device) -> torch.Tensor:
    """One uniformly-random complementary label per sample (!= true label),
    matching the paper's Sec 2.2 generation exactly (not this repo's usual
    k-sized-candidate-set complement). Returns a [B, 1] long tensor (the
    format every CLL loss module here already expects: [B, max_m], -1 =
    padding; max_m=1 needs no padding)."""
    rng = np.random.RandomState(seed_val)
    tl = true_labels.cpu().numpy()
    comp = np.empty((len(tl), 1), dtype=np.int64)
    for i, y in enumerate(tl):
        choices = np.delete(np.arange(C), int(y))
        comp[i, 0] = rng.choice(choices)
    return torch.from_numpy(comp).to(device)


def _loss_grad(loss_fn, logits: torch.Tensor, comp_labels: torch.Tensor) -> torch.Tensor:
    leaf = logits.detach().clone().requires_grad_(True)
    loss = loss_fn(leaf, comp_labels)
    loss.backward()
    return leaf.grad.detach()


def _append_row(out_path: str, row: dict):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new_file = not os.path.isfile(out_path)
    with open(out_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        w.writerow(row)


def log_checkpoint(loss_fns: dict, logits: torch.Tensor, true_labels: torch.Tensor,
                    dataset: str, C: int, epoch: int, seed: int, m_resamples: int,
                    out_path: str, device):
    """`loss_fns`: {algorithm_name: loss_module}. One CSV row per algorithm
    per checkpoint. `logits` are the frozen reference model's cached output
    on the fixed eval batch (no_grad, computed once by the caller) --
    identical for every algorithm and every resample, exactly as in the
    paper (only the resampled complementary label changes between resamples;
    the model/forward pass never changes)."""
    oracle_grad = oracle_cls_grad(logits, true_labels)
    for algo, loss_fn in loss_fns.items():
        grads = []
        for m in range(m_resamples):
            seed_val = hash((dataset, C, epoch, algo, m)) & 0xFFFFFFFF
            comp_labels = resample_single_cl_batch(true_labels, C, seed_val, device)
            grads.append(_loss_grad(loss_fn, logits, comp_labels))
        stats = _reduce(grads, oracle_grad)
        _append_row(out_path, dict(dataset=dataset, C=C, algorithm=algo, seed=seed, epoch=epoch,
                                    **{k: round(v, 8) for k, v in stats.items()}))
