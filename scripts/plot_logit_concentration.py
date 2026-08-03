"""
Compare logit concentration (PiCO vs ComCo) across epochs for k = 5, 10, 15.

Metrics plotted (two rows):
  Row 0 — Mean softmax entropy         (lower  = more concentrated / confident)
  Row 1 — Mean max-softmax probability (higher = more concentrated / confident)

Output: plots/extended_analysis/logit_concentration.png

Usage:
    python scripts/plot_logit_concentration.py [--base_dir results/extended_analysis]
                                               [--out_dir  plots/extended_analysis]
                                               [--C 20] [--k 5 10 15]
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Defaults ──────────────────────────────────────────────────────────────────

ALGS   = ['PiCO', 'ComCo']
KS_DEF = [5, 10, 15]
COLORS = {'PiCO': '#9467bd', 'ComCo': '#8c564b'}
PROT_START = 80   # PiCO warmup boundary

# ── I/O helpers ───────────────────────────────────────────────────────────────

def _epoch_csv(base, alg, C, k, epoch):
    return os.path.join(base, alg, f'C{C}_k{k}', 'logits', f'ep{epoch:04d}.csv')


def available_epochs(base, alg, C, k):
    d = os.path.join(base, alg, f'C{C}_k{k}', 'logits')
    if not os.path.isdir(d):
        return []
    out = []
    for fname in sorted(os.listdir(d)):
        if fname.startswith('ep') and fname.endswith('.csv') and len(fname) == 11:
            try:
                out.append(int(fname[2:6]))
            except ValueError:
                pass
    return out


def load_logits(path, C):
    """Return float32 array [N, C], or None if file missing/empty."""
    if not os.path.isfile(path):
        return None
    cols = [f'logit_{c}' for c in range(C)]
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append([float(row[c]) for c in cols])
    return np.array(rows, dtype=np.float32) if rows else None

# ── Metric computation ────────────────────────────────────────────────────────

def _softmax(x):
    """x: [N, C] → [N, C] softmax (numerically stable)."""
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def _entropy(probs):
    """Mean Shannon entropy of rows. probs: [N, C]."""
    return -(probs * np.log(probs + 1e-12)).sum(axis=1).mean()


def _max_prob(probs):
    """Mean max probability of rows. probs: [N, C]."""
    return probs.max(axis=1).mean()


def compute_metrics(base, alg, C, k):
    """
    Returns (epochs, entropies, max_probs) — parallel lists.
    Skips epochs with missing files silently.
    """
    epochs = available_epochs(base, alg, C, k)
    if not epochs:
        return [], [], []

    ents, mps, valid_eps = [], [], []
    for ep in epochs:
        logits = load_logits(_epoch_csv(base, alg, C, k, ep), C)
        if logits is None:
            continue
        probs = _softmax(logits)
        ents.append(_entropy(probs))
        mps.append(_max_prob(probs))
        valid_eps.append(ep)

    return valid_eps, ents, mps

# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(base, out_dir, C, ks, algs):
    n_k    = len(ks)
    fig, axes = plt.subplots(2, n_k,
                             figsize=(5 * n_k, 8),
                             sharey='row')          # same y-scale within each metric row

    fig.suptitle(f'Logit Concentration: PiCO vs ComCo  (C={C})',
                 fontsize=13, fontweight='bold')

    ROW_LABELS = [
        'Mean Softmax Entropy\n(↓ = more concentrated)',
        'Mean Max-Softmax Prob\n(↑ = more concentrated)',
    ]

    for col, k in enumerate(ks):
        ax_ent  = axes[0, col]
        ax_prob = axes[1, col]

        has_data = False
        for alg in algs:
            eps, ents, mps = compute_metrics(base, alg, C, k)
            if not eps:
                print(f'  [skip] {alg}  C={C}  k={k} — no logit files found')
                continue
            has_data = True
            color = COLORS.get(alg, None)
            lw = 1.8
            ax_ent.plot(eps, ents, color=color, label=alg, linewidth=lw)
            ax_prob.plot(eps, mps,  color=color, label=alg, linewidth=lw)

        # Warmup boundary
        for ax in (ax_ent, ax_prob):
            ax.axvline(PROT_START, color='gray', linestyle='--',
                       linewidth=0.9, alpha=0.7, label=f'warmup ({PROT_START})')
            ax.set_xlabel('Epoch', fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc='best')

        ax_ent.set_title(f'k = {k}', fontsize=11)

        # Row labels only on leftmost column
        if col == 0:
            ax_ent.set_ylabel(ROW_LABELS[0], fontsize=8)
            ax_prob.set_ylabel(ROW_LABELS[1], fontsize=8)

        if not has_data:
            for ax in (ax_ent, ax_prob):
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes,
                        ha='center', va='center', color='gray', fontsize=10)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'logit_concentration.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out_path}')

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='results/extended_analysis')
    parser.add_argument('--out_dir',  default='plots/extended_analysis')
    parser.add_argument('--C',        type=int,   default=20)
    parser.add_argument('--k',        type=int,   nargs='+', default=KS_DEF)
    parser.add_argument('--alg',      nargs='+',  default=ALGS,
                        help='Algorithms to include (default: PiCO ComCo)')
    args = parser.parse_args()

    print(f'Base : {args.base_dir}')
    print(f'Algs : {args.alg}')
    print(f'C={args.C}   k={args.k}')
    print()
    plot(args.base_dir, args.out_dir, args.C, args.k, args.alg)


if __name__ == '__main__':
    main()
