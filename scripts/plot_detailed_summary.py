"""
Generate all comparison plots from detailed_analysis results.

Plots saved to plots/detailed_summary/:
  A1  final_acc_vs_k.png         — Final accuracy vs k (4 methods)
  A2  learning_curves.png        — Learning curves at k=5, 10, 15
  B1  loss_components.png        — cls / cont loss at k=5 and k=15
  B2  cls_ratio.png              — cls_ratio over epochs (all k per method)
  C1  prediction_entropy.png     — Prediction distribution entropy vs k
  C2  effective_classes.png      — # effectively predicted classes vs k
  D2  perclass_std.png           — Per-class accuracy std vs k
  D3  confusion_k5.png           — Confusion matrices at k=5
      confusion_k15.png          — Confusion matrices at k=15
  E1  accuracy_trajectory.png    — Accuracy heatmap (k × epoch)

Usage:
  python scripts/plot_detailed_summary.py
  python scripts/plot_detailed_summary.py --show_class_names
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants ──────────────────────────────────────────────────────────────────

C    = 20
KS   = list(range(5, 16))   # 5 … 15
ALGS = ['PiCO', 'PiCO-CLS', 'PiCO-SC', 'ComCo']

COLORS  = {'PiCO': '#9467bd', 'PiCO-CLS': '#e377c2', 'PiCO-SC': '#d4a800', 'ComCo': '#8c564b'}
MARKERS = {'PiCO': 's',       'PiCO-CLS': '*',        'PiCO-SC': 'h',       'ComCo': '^'}
LS      = {'PiCO': '-',       'PiCO-CLS': '-',         'PiCO-SC': '--',      'ComCo': '-'}

BASE = 'results/detailed_analysis'
OUT  = 'plots/detailed_summary'


# ── Data readers ───────────────────────────────────────────────────────────────

def _path(alg, k, fname):
    return os.path.join(BASE, alg, f'C{C}_k{k}', fname)


def read_loss_curve(alg, k):
    """
    Returns dict with numpy arrays or None.
    Handles 5-column CSVs (no cls_ratio) by computing the ratio from cls/total.
    Columns: epoch, cls_loss, cont_loss, total_loss, [cls_ratio, wcont_ratio,] overall_acc
    """
    p = _path(alg, k, 'loss_curve.csv')
    if not os.path.isfile(p):
        return None
    rows = []
    with open(p, newline='') as f:
        reader = csv.reader(f)
        next(reader)  # skip header row
        for row in reader:
            if row:
                rows.append([float(v) for v in row])
    if not rows:
        return None
    # Pad inhomogeneous rows to max length (some early rows may lack cls_ratio cols)
    max_cols = max(len(r) for r in rows)
    padded = [r + [float('nan')] * (max_cols - len(r)) for r in rows]
    arr = np.array(padded)
    d = {
        'epoch':       arr[:, 0].astype(int),
        'cls_loss':    arr[:, 1],
        'cont_loss':   arr[:, 2],
        'total_loss':  arr[:, 3],
        'overall_acc': arr[:, -1],
    }
    if arr.shape[1] == 7:
        d['cls_ratio']   = arr[:, 4]
        d['wcont_ratio'] = arr[:, 5]
    else:
        # Compute ratio from available columns
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.where(d['total_loss'] > 0,
                             d['cls_loss'] / d['total_loss'], 1.0)
        d['cls_ratio']   = ratio
        d['wcont_ratio'] = 1.0 - ratio
    return d


def read_per_class(alg, k):
    p = _path(alg, k, 'per_class_loss.csv')
    if not os.path.isfile(p):
        return None
    rows = []
    with open(p, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return None
    epochs  = [int(r['epoch']) for r in rows]
    acc_mat = np.array([[float(r.get(f'acc_class_{c}', 'nan')) for c in range(C)]
                         for r in rows])
    overall = np.array([float(r['overall_acc']) for r in rows])
    return {'epochs': epochs, 'acc_mat': acc_mat, 'overall': overall}


def read_logits_final(alg, k):
    """Load predictions from the last available logits checkpoint."""
    logit_dir = _path(alg, k, 'logits')
    if not os.path.isdir(logit_dir):
        return None
    files = sorted(Path(logit_dir).glob('ep*.csv'))
    if not files:
        return None
    true_labels, pred_labels = [], []
    with open(files[-1], newline='') as f:
        for row in csv.DictReader(f):
            true_labels.append(int(row['true_label']))
            pred_labels.append(int(row['pred_label']))
    return {
        'true':  np.array(true_labels),
        'pred':  np.array(pred_labels),
        'epoch': int(files[-1].stem[2:]),
    }


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


def _add_legend(ax, **kw):
    ax.legend(fontsize=8, **kw)


def _alg_line(ax, alg, xs, ys, **kw):
    ax.plot(xs, ys,
            color=COLORS[alg], marker=MARKERS[alg],
            linestyle=LS[alg], linewidth=1.8, markersize=7,
            label=alg, **kw)


# ── A1: Final accuracy vs k ────────────────────────────────────────────────────

def plot_A1():
    print('A1  Final accuracy vs k')
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in ALGS:
        xs, ys = [], []
        for k in KS:
            d = read_loss_curve(alg, k)
            if d is not None:
                xs.append(k)
                ys.append(d['overall_acc'][-1])
        _alg_line(ax, alg, xs, ys)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=11)
    ax.set_ylabel('Final Overall Accuracy (%)', fontsize=11)
    ax.set_title('Final Accuracy vs k  —  C=20, epoch 200', fontsize=12)
    ax.set_xticks(KS)
    ax.grid(alpha=0.3)
    _add_legend(ax)
    _save(fig, 'A1_final_acc_vs_k.png')


# ── A2: Learning curves at k=5, 10, 15 ───────────────────────────────────────

def plot_A2():
    print('A2  Learning curves')
    sel = [5, 10, 15]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, k in zip(axes, sel):
        for alg in ALGS:
            d = read_loss_curve(alg, k)
            if d is not None:
                ax.plot(d['epoch'], d['overall_acc'],
                        color=COLORS[alg], linestyle=LS[alg],
                        label=alg, linewidth=1.5, alpha=0.9)
        ax.set_title(f'k = {k}', fontsize=11)
        ax.set_xlabel('Epoch', fontsize=9)
        ax.set_ylabel('Overall Accuracy (%)', fontsize=9)
        ax.set_xlim(1, 200)
        ax.grid(alpha=0.3)
        _add_legend(ax, loc='lower right')
    fig.suptitle('Learning Curves  —  C=20', fontsize=13)
    fig.tight_layout()
    _save(fig, 'A2_learning_curves.png')


# ── B1: cls / cont loss at k=5 and k=15 ──────────────────────────────────────

def plot_B1():
    print('B1  Loss components')
    sel = [5, 15]
    n_rows, n_cols = len(sel), len(ALGS)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3.5 * n_rows),
                              sharex=True)
    for r, k in enumerate(sel):
        for c, alg in enumerate(ALGS):
            ax = axes[r][c]
            d = read_loss_curve(alg, k)
            if d is not None:
                ax.plot(d['epoch'], d['cls_loss'],
                        color='steelblue', linewidth=1.3, label='cls')
                ax.plot(d['epoch'], d['cont_loss'],
                        color='darkorange', linewidth=1.3, label='cont')
            ax.set_title(f'{alg}  k={k}', fontsize=9)
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel('Loss', fontsize=8)
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle('cls / cont Loss Components  —  C=20', fontsize=13)
    fig.tight_layout()
    _save(fig, 'B1_loss_components.png')


# ── B2: cls_ratio vs epoch, all k on same plot per method ────────────────────

def plot_B2():
    print('B2  cls_ratio')
    cmap = plt.cm.plasma
    fig, axes = plt.subplots(1, len(ALGS), figsize=(4.5 * len(ALGS), 4), sharey=True)
    for ax, alg in zip(axes, ALGS):
        for i, k in enumerate(KS):
            d = read_loss_curve(alg, k)
            if d is not None:
                color = cmap(i / max(len(KS) - 1, 1))
                ax.plot(d['epoch'], d['cls_ratio'],
                        color=color, linewidth=1.0, alpha=0.85, label=f'k={k}')
        ax.set_title(alg, fontsize=10)
        ax.set_xlabel('Epoch', fontsize=8)
        ax.set_ylim(-0.05, 1.1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('cls_ratio  (cls / total loss)', fontsize=9)
    # colorbar-like legend on last subplot
    handles = [plt.Line2D([0], [0], color=cmap(i / max(len(KS)-1, 1)),
                           linewidth=1.5, label=f'k={k}')
               for i, k in enumerate(KS)]
    axes[-1].legend(handles=handles, fontsize=6, loc='upper right',
                    ncol=2, title='k', title_fontsize=7)
    fig.suptitle('cls Loss Ratio vs Epoch  —  C=20', fontsize=13)
    fig.tight_layout()
    _save(fig, 'B2_cls_ratio.png')


# ── C1: Prediction distribution entropy vs k ─────────────────────────────────

def _entropy(pred):
    counts = np.bincount(pred, minlength=C).astype(float)
    probs  = counts / counts.sum()
    probs  = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


def plot_C1():
    print('C1  Prediction entropy')
    max_h = float(np.log(C))
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in ALGS:
        xs, ys = [], []
        for k in KS:
            d = read_logits_final(alg, k)
            if d is not None:
                xs.append(k)
                ys.append(_entropy(d['pred']))
        _alg_line(ax, alg, xs, ys)
    ax.axhline(max_h, color='gray', linestyle='--', linewidth=1,
               label=f'uniform  log({C}) ≈ {max_h:.2f}')
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=11)
    ax.set_ylabel('H(prediction distribution)', fontsize=11)
    ax.set_title('Prediction Collapse Metric (Entropy)  —  C=20, final epoch', fontsize=12)
    ax.set_xticks(KS)
    ax.set_ylim(0, max_h * 1.15)
    ax.grid(alpha=0.3)
    _add_legend(ax)
    _save(fig, 'C1_prediction_entropy.png')


# ── C2: Effective predicted classes vs k ─────────────────────────────────────

def _effective_classes(pred, pct_thresh=5.0):
    counts = np.bincount(pred, minlength=C).astype(float)
    pct    = counts / counts.sum() * 100
    return int((pct > pct_thresh).sum())


def plot_C2():
    print('C2  Effective predicted classes')
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in ALGS:
        xs, ys = [], []
        for k in KS:
            d = read_logits_final(alg, k)
            if d is not None:
                xs.append(k)
                ys.append(_effective_classes(d['pred']))
        _alg_line(ax, alg, xs, ys)
    ax.axhline(C, color='gray', linestyle='--', linewidth=1, label=f'max = {C}')
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=11)
    ax.set_ylabel('# classes with >5% prediction share', fontsize=11)
    ax.set_title('Effective Predicted Classes  —  C=20, final epoch', fontsize=12)
    ax.set_xticks(KS)
    ax.set_ylim(0, C + 2)
    ax.grid(alpha=0.3)
    _add_legend(ax)
    _save(fig, 'C2_effective_classes.png')


# ── D2: Per-class accuracy std vs k ──────────────────────────────────────────

def plot_D2():
    print('D2  Per-class accuracy std')
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in ALGS:
        xs, ys = [], []
        for k in KS:
            d = read_per_class(alg, k)
            if d is not None:
                xs.append(k)
                ys.append(float(np.nanstd(d['acc_mat'][-1])))
        _alg_line(ax, alg, xs, ys)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=11)
    ax.set_ylabel('Std of per-class accuracy (%)', fontsize=11)
    ax.set_title('Per-class Accuracy Imbalance  —  C=20, final epoch', fontsize=12)
    ax.set_xticks(KS)
    ax.grid(alpha=0.3)
    _add_legend(ax)
    _save(fig, 'D2_perclass_std.png')


# ── D3: Confusion matrices at k=5 and k=15 ───────────────────────────────────

def _confusion_matrix(true, pred):
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm


def _plot_confusion_for_k(k, class_names=None):
    fig, axes = plt.subplots(1, len(ALGS), figsize=(4.5 * len(ALGS), 4.5))
    for ax, alg in zip(axes, ALGS):
        d = read_logits_final(alg, k)
        if d is not None:
            cm       = _confusion_matrix(d['true'], d['pred'])
            row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
            cm_norm  = cm / row_sums   # row-normalized = per-class recall
            im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'{alg}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True', fontsize=8)
        ticks = range(C)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        if class_names:
            ax.set_xticklabels(class_names, rotation=90, fontsize=5)
            ax.set_yticklabels(class_names, fontsize=5)
        else:
            ax.set_xticklabels(list(ticks), fontsize=6)
            ax.set_yticklabels(list(ticks), fontsize=6)
    epoch_label = ''
    if d is not None:
        epoch_label = f', ep {d["epoch"]}'
    fig.suptitle(f'Confusion Matrix (row-norm = recall)  —  C=20  k={k}{epoch_label}',
                 fontsize=12)
    fig.tight_layout()
    _save(fig, f'D3_confusion_k{k}.png')


def plot_D3(class_names=None):
    print('D3  Confusion matrices')
    for k in [5, 15]:
        _plot_confusion_for_k(k, class_names)


# ── E1: Accuracy trajectory heatmap (X=k, Y=epoch) ───────────────────────────

def plot_E1():
    print('E1  Accuracy trajectory heatmap')
    fig, axes = plt.subplots(1, len(ALGS), figsize=(4 * len(ALGS), 6), sharey=True)
    im_ref = None
    for ax, alg in zip(axes, ALGS):
        # Collect columns [T_i] for each k, then pad to uniform length
        epoch_lists = []
        overalls    = []
        for k in KS:
            d = read_per_class(alg, k)
            if d is not None:
                epoch_lists.append(d['epochs'])
                overalls.append(list(d['overall']))
            else:
                epoch_lists.append([])
                overalls.append([])

        max_len   = max((len(o) for o in overalls), default=0)
        ref_epochs = next((e for e in epoch_lists if e), list(range(10, 10*max_len+1, 10)))

        mat = np.full((max_len, len(KS)), np.nan)
        for i, col in enumerate(overalls):
            mat[:len(col), i] = col

        im_ref = ax.imshow(mat, aspect='auto', origin='lower',
                           cmap='RdYlGn', vmin=0, vmax=100,
                           interpolation='nearest')
        ax.set_title(alg, fontsize=10, fontweight='bold')
        ax.set_xlabel('k', fontsize=9)
        ax.set_xticks(range(len(KS)))
        ax.set_xticklabels(KS, fontsize=7)
        if ax is axes[0]:
            ax.set_ylabel('Epoch checkpoint', fontsize=9)
            ax.set_yticks(range(len(ref_epochs)))
            ax.set_yticklabels(ref_epochs, fontsize=6)

    if im_ref is not None:
        fig.colorbar(im_ref, ax=list(axes), label='Overall Accuracy (%)',
                     shrink=0.7, pad=0.02)
    fig.suptitle('Accuracy Trajectory Heatmap  —  C=20', fontsize=13)
    _save(fig, 'E1_accuracy_trajectory.png')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_class_names', action='store_true',
                        help='Show CIFAR-100 class names on confusion matrix axes')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--only',     nargs='+', default=None,
                        help='Run only listed plots, e.g. --only A1 C1 E1')
    args = parser.parse_args()

    class_names = None
    if args.show_class_names:
        try:
            from src.cifar100_subset import select_cifar100_classes
            from torchvision.datasets import CIFAR100
            idxs = select_cifar100_classes(C, seed=args.seed)
            ds   = CIFAR100(root=args.data_dir, train=True, download=True)
            class_names = [ds.classes[i] for i in idxs]
        except Exception as e:
            print(f'[warn] Could not load class names: {e}')

    only = set(args.only) if args.only else None

    def run(tag, fn, *a, **kw):
        if only and tag not in only:
            return
        fn(*a, **kw)

    run('A1', plot_A1)
    run('A2', plot_A2)
    run('B1', plot_B1)
    run('B2', plot_B2)
    run('C1', plot_C1)
    run('C2', plot_C2)
    run('D2', plot_D2)
    run('D3', plot_D3, class_names)
    run('E1', plot_E1)

    print(f'\nAll done — output: {OUT}/')


if __name__ == '__main__':
    main()
