"""
Generate all plots for the extended analysis (5 methods × k∈{5,10,15,19} × 500 epochs).

Per-run plots (same as plots/detailed_analysis/, saved per alg/k):
  plots/extended_analysis/{alg}/C20_k{k}/
    loss_curve.png              — cls / cont / total loss over epochs
    per_class_loss_heatmap.png  — per-class CE loss heatmap (epoch × class)
    per_class_acc_heatmap.png   — per-class accuracy heatmap (epoch × class)
    logit_confidence_heatmap.png — mean softmax confidence on true class

Comparison plots (same structure as plots/detailed_summary/):
  plots/extended_summary/
    A1  final_acc_vs_k.png
    A2  learning_curves.png
    B1  loss_components.png
    B2  cls_ratio.png
    C1  prediction_entropy.png
    C2  effective_classes.png
    D2  perclass_std.png
    D3  confusion_k5.png / confusion_k19.png
    E1  accuracy_trajectory.png
    E2  E2_class{c:02d}_{name}.png  (20 figures)

Missing data is skipped silently.

Usage:
  python scripts/plot_extended_summary.py
  python scripts/plot_extended_summary.py --show_class_names
  python scripts/plot_extended_summary.py --only A1 C1 E1
  python scripts/plot_extended_summary.py --only per_run
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
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Constants ──────────────────────────────────────────────────────────────────

C    = 20
KS   = [5, 10, 15, 19]
ALGS = ['PiCO', 'PiCO-Uniform', 'PiCO-CLS', 'PiCO-SC', 'ComCo']

COLORS  = {
    'PiCO':         '#9467bd',
    'PiCO-Uniform': '#ff7f0e',
    'PiCO-CLS':     '#e377c2',
    'PiCO-SC':      '#d4a800',
    'ComCo':        '#8c564b',
}
MARKERS = {
    'PiCO':         's',
    'PiCO-Uniform': 'D',
    'PiCO-CLS':     '*',
    'PiCO-SC':      'h',
    'ComCo':        '^',
}
LS = {
    'PiCO':         '-',
    'PiCO-Uniform': '--',
    'PiCO-CLS':     '-',
    'PiCO-SC':      '--',
    'ComCo':        '-',
}

BASE     = 'results/extended_analysis'
OUT_RUN  = 'plots/extended_analysis'   # per-run plots
OUT_CMP  = 'plots/extended_summary'    # comparison plots


# ── Data readers ───────────────────────────────────────────────────────────────

def _path(alg, k, fname):
    return os.path.join(BASE, alg, f'C{C}_k{k}', fname)


def read_loss_curve(alg, k):
    p = _path(alg, k, 'loss_curve.csv')
    if not os.path.isfile(p):
        return None
    rows = []
    with open(p, newline='') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row:
                rows.append([float(v) for v in row])
    if not rows:
        return None
    max_cols = max(len(r) for r in rows)
    arr = np.array([r + [float('nan')] * (max_cols - len(r)) for r in rows])
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
        with np.errstate(divide='ignore', invalid='ignore'):
            r = np.where(d['total_loss'] > 0, d['cls_loss'] / d['total_loss'], 1.0)
        d['cls_ratio']   = r
        d['wcont_ratio'] = 1.0 - r
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
    loss_mat = np.array([[float(r.get(f'loss_class_{c}', 'nan')) for c in range(C)]
                          for r in rows])
    overall = np.array([float(r['overall_acc']) for r in rows])
    return {'epochs': epochs, 'acc_mat': acc_mat, 'loss_mat': loss_mat, 'overall': overall}


def read_logits_final(alg, k):
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


def read_logits_all(alg, k):
    """Returns list of (epoch, true, pred, logits[N,C]) for all checkpoints."""
    logit_dir = _path(alg, k, 'logits')
    if not os.path.isdir(logit_dir):
        return []
    results = []
    for fpath in sorted(Path(logit_dir).glob('ep*.csv')):
        ep = int(fpath.stem[2:])
        true_l, pred_l, logit_l = [], [], []
        with open(fpath, newline='') as f:
            for row in csv.DictReader(f):
                true_l.append(int(row['true_label']))
                pred_l.append(int(row['pred_label']))
                logit_l.append([float(row[f'logit_{c}']) for c in range(C)])
        results.append((ep, np.array(true_l), np.array(pred_l), np.array(logit_l)))
    return results


# ── Save helpers ───────────────────────────────────────────────────────────────

def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  → {p}')


def _alg_line(ax, alg, xs, ys, **kw):
    ax.plot(xs, ys, color=COLORS[alg], marker=MARKERS[alg],
            linestyle=LS[alg], linewidth=1.8, markersize=7,
            label=alg, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# PER-RUN PLOTS  (one set per alg × k)
# ══════════════════════════════════════════════════════════════════════════════

def _per_run_loss_curve(alg, k):
    d = read_loss_curve(alg, k)
    if d is None:
        return
    plots_dir = os.path.join(OUT_RUN, alg, f'C{C}_k{k}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
    fig.suptitle(f'{alg}  C={C}  k={k}  —  Loss Curves', fontsize=12)

    ax1.plot(d['epoch'], d['cls_loss'],   label='cls',   color='#1f77b4', linewidth=1.5)
    ax1.plot(d['epoch'], d['cont_loss'],  label='cont',  color='#ff7f0e', linewidth=1.5)
    ax1.plot(d['epoch'], d['total_loss'], label='total', color='#2ca02c', linewidth=2, linestyle='--')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Absolute loss'); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    ax2.stackplot(d['epoch'], d['cls_ratio'], d['wcont_ratio'],
                  labels=['cls / total', 'w·cont / total'],
                  colors=['#1f77b4', '#ff7f0e'], alpha=0.75)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Ratio')
    ax2.set_ylim(0, 1); ax2.set_title('Weighted loss ratio')
    ax2.legend(fontsize=9, loc='lower right'); ax2.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, plots_dir, 'loss_curve.png')


def _per_run_per_class_loss_heatmap(alg, k, class_names=None):
    d = read_per_class(alg, k)
    if d is None:
        return
    plots_dir   = os.path.join(OUT_RUN, alg, f'C{C}_k{k}')
    epochs_list = d['epochs']
    loss_mat    = d['loss_mat']

    fig, ax = plt.subplots(figsize=(max(8, len(epochs_list) // 2), 6))
    im = ax.imshow(loss_mat.T, aspect='auto', origin='lower',
                   cmap='RdYlGn_r', interpolation='nearest')
    ax.set_xticks(range(len(epochs_list)))
    ax.set_xticklabels(epochs_list, rotation=45, fontsize=7)
    ax.set_xlabel('Epoch checkpoint', fontsize=9)
    ax.set_yticks(range(C))
    ax.set_yticklabels(class_names if class_names else range(C), fontsize=7)
    ax.set_ylabel('Class', fontsize=9)
    ax.set_title(f'{alg}  C={C}  k={k}  —  Per-class CE Loss (test)', fontsize=11)
    plt.colorbar(im, ax=ax, label='CE loss')
    fig.tight_layout()
    _save(fig, plots_dir, 'per_class_loss_heatmap.png')


def _per_run_per_class_acc_heatmap(alg, k, class_names=None):
    d = read_per_class(alg, k)
    if d is None:
        return
    plots_dir   = os.path.join(OUT_RUN, alg, f'C{C}_k{k}')
    epochs_list = d['epochs']
    acc_mat     = d['acc_mat']

    fig, ax = plt.subplots(figsize=(max(8, len(epochs_list) // 2), 6))
    im = ax.imshow(acc_mat.T, aspect='auto', origin='lower',
                   cmap='RdYlGn', vmin=0, vmax=100, interpolation='nearest')
    ax.set_xticks(range(len(epochs_list)))
    ax.set_xticklabels(epochs_list, rotation=45, fontsize=7)
    ax.set_xlabel('Epoch checkpoint', fontsize=9)
    ax.set_yticks(range(C))
    ax.set_yticklabels(class_names if class_names else range(C), fontsize=7)
    ax.set_ylabel('Class', fontsize=9)
    ax.set_title(f'{alg}  C={C}  k={k}  —  Per-class Accuracy (test)', fontsize=11)
    plt.colorbar(im, ax=ax, label='Accuracy (%)')
    fig.tight_layout()
    _save(fig, plots_dir, 'per_class_acc_heatmap.png')


def _per_run_logit_confidence(alg, k, class_names=None):
    checkpoints = read_logits_all(alg, k)
    if not checkpoints:
        return
    plots_dir   = os.path.join(OUT_RUN, alg, f'C{C}_k{k}')
    epochs_list = []
    conf_mat    = []

    for ep, true_l, pred_l, logits in checkpoints:
        epochs_list.append(ep)
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        row = []
        for c in range(C):
            mask = (true_l == c)
            row.append(probs[mask, c].mean() if mask.sum() > 0 else float('nan'))
        conf_mat.append(row)

    conf_mat = np.array(conf_mat)   # [T, C]

    fig, ax = plt.subplots(figsize=(max(8, len(epochs_list) // 2), 6))
    im = ax.imshow(conf_mat.T, aspect='auto', origin='lower',
                   cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
    ax.set_xticks(range(len(epochs_list)))
    ax.set_xticklabels(epochs_list, rotation=45, fontsize=7)
    ax.set_xlabel('Epoch checkpoint', fontsize=9)
    ax.set_yticks(range(C))
    ax.set_yticklabels(class_names if class_names else range(C), fontsize=7)
    ax.set_ylabel('Class', fontsize=9)
    ax.set_title(f'{alg}  C={C}  k={k}  —  Mean softmax(true class) per class', fontsize=11)
    plt.colorbar(im, ax=ax, label='Mean confidence on true class')
    fig.tight_layout()
    _save(fig, plots_dir, 'logit_confidence_heatmap.png')


def plot_per_run(class_names=None):
    print('Per-run plots')
    for alg in ALGS:
        for k in KS:
            if not os.path.isdir(os.path.join(BASE, alg, f'C{C}_k{k}')):
                continue
            print(f'  {alg} k={k}')
            _per_run_loss_curve(alg, k)
            _per_run_per_class_loss_heatmap(alg, k, class_names)
            _per_run_per_class_acc_heatmap(alg, k, class_names)
            _per_run_logit_confidence(alg, k, class_names)


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON PLOTS  (across algs / k values)
# ══════════════════════════════════════════════════════════════════════════════

# ── A1: Final accuracy vs k ────────────────────────────────────────────────────

def _acc_at_epoch(d, at_epoch=None):
    """Return accuracy at the closest checkpoint <= at_epoch. None = last."""
    if at_epoch is None:
        return float(d['overall'][-1])
    epochs = np.array(d['epochs'])
    mask = epochs <= at_epoch
    if not mask.any():
        return None
    idx = np.where(mask)[0][-1]
    return float(d['overall'][idx])


def _plot_A1_single(ax, at_epoch=None, algs=None):
    if algs is None:
        algs = ALGS
    label_ep = f'ep {at_epoch}' if at_epoch else 'final'
    for alg in algs:
        xs, ys = [], []
        for k in KS:
            d = read_per_class(alg, k)
            if d is not None:
                acc = _acc_at_epoch(d, at_epoch)
                if acc is not None:
                    xs.append(k)
                    ys.append(acc)
        if xs:
            _alg_line(ax, alg, xs, ys)
    ax.set_xlabel('k  (# partial labels)', fontsize=11)
    ax.set_ylabel(f'Accuracy @ {label_ep} (%)', fontsize=11)
    ax.set_xticks(KS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)


def plot_A1():
    print('A1  Final accuracy vs k  (+ep200 version)')

    # full (final epoch)
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_A1_single(ax, at_epoch=None)
    ax.set_title('Final Accuracy vs k  —  C=20, epoch 500', fontsize=12)
    _save(fig, OUT_CMP, 'A1_final_acc_vs_k.png')

    # epoch-200 snapshot
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_A1_single(ax, at_epoch=200)
    ax.set_title('Accuracy @ ep200 vs k  —  C=20', fontsize=12)
    _save(fig, OUT_CMP, 'A1_ep200_acc_vs_k.png')

    # without ComCo
    no_comco = [a for a in ALGS if a != 'ComCo']
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_A1_single(ax, at_epoch=None, algs=no_comco)
    ax.set_title('Final Accuracy vs k  —  C=20, epoch 500  (PLL methods only)', fontsize=12)
    _save(fig, OUT_CMP, 'A1_final_acc_vs_k_no_comco.png')


# ── A2: Learning curves ────────────────────────────────────────────────────────

def plot_A2():
    print('A2  Learning curves')
    fig, axes = plt.subplots(1, len(KS), figsize=(5 * len(KS), 5), sharey=True)

    # collect global acc range for explicit ylim
    all_acc = []
    for alg in ALGS:
        for k in KS:
            d = read_per_class(alg, k)
            if d is not None:
                all_acc.extend(d['overall'].tolist())
    ymin = max(0,   min(all_acc) - 3) if all_acc else 0
    ymax = min(100, max(all_acc) + 3) if all_acc else 100

    for i, (ax, k) in enumerate(zip(axes, KS)):
        for alg in ALGS:
            d = read_per_class(alg, k)
            if d is not None:
                ax.plot(d['epochs'], d['overall'],
                        color=COLORS[alg], linestyle=LS[alg],
                        marker=MARKERS[alg], markersize=3,
                        label=alg, linewidth=1.5, alpha=0.9)
        ax.set_title(f'k = {k}', fontsize=11)
        ax.set_xlabel('Epoch', fontsize=9)
        if i == 0:
            ax.set_ylabel('Overall Accuracy (%)', fontsize=9)
        ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc='lower right')
    fig.suptitle('Learning Curves  —  C=20', fontsize=13)
    fig.tight_layout()
    _save(fig, OUT_CMP, 'A2_learning_curves.png')


# ── B1: cls / cont loss at k=5 and k=19 ──────────────────────────────────────

def plot_B1():
    print('B1  Loss components')
    sel = [KS[0], KS[-1]]   # k=5 and k=19
    fig, axes = plt.subplots(len(sel), len(ALGS),
                              figsize=(4 * len(ALGS), 3.5 * len(sel)),
                              sharex=True)
    for r, k in enumerate(sel):
        for c, alg in enumerate(ALGS):
            ax = axes[r][c]
            d  = read_loss_curve(alg, k)
            if d is not None:
                ax.plot(d['epoch'], d['cls_loss'],  color='steelblue',  linewidth=1.3, label='cls')
                ax.plot(d['epoch'], d['cont_loss'], color='darkorange', linewidth=1.3, label='cont')
            ax.set_title(f'{alg}  k={k}', fontsize=9)
            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel('Loss', fontsize=8)
            ax.grid(alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle('cls / cont Loss Components  —  C=20', fontsize=13)
    fig.tight_layout()
    _save(fig, OUT_CMP, 'B1_loss_components.png')


# ── B2: cls_ratio vs epoch ────────────────────────────────────────────────────

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
    handles = [plt.Line2D([0], [0], color=cmap(i / max(len(KS)-1, 1)),
                           linewidth=1.5, label=f'k={k}')
               for i, k in enumerate(KS)]
    axes[-1].legend(handles=handles, fontsize=8, loc='upper right', title='k')
    fig.suptitle('cls Loss Ratio vs Epoch  —  C=20', fontsize=13)
    fig.tight_layout()
    _save(fig, OUT_CMP, 'B2_cls_ratio.png')


# ── C1: Prediction entropy vs k ───────────────────────────────────────────────

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
        if xs:
            _alg_line(ax, alg, xs, ys)
    ax.axhline(max_h, color='gray', linestyle='--', linewidth=1,
               label=f'uniform  log({C}) ≈ {max_h:.2f}')
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('H(prediction distribution)', fontsize=11)
    ax.set_title('Prediction Collapse Metric (Entropy)  —  C=20, final epoch', fontsize=12)
    ax.set_xticks(KS)
    ax.set_ylim(0, max_h * 1.15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    _save(fig, OUT_CMP, 'C1_prediction_entropy.png')


# ── C2: Effective predicted classes vs k ─────────────────────────────────────

def _effective_classes(pred, pct=5.0):
    counts = np.bincount(pred, minlength=C).astype(float)
    return int((counts / counts.sum() * 100 > pct).sum())


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
        if xs:
            _alg_line(ax, alg, xs, ys)
    ax.axhline(C, color='gray', linestyle='--', linewidth=1, label=f'max = {C}')
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('# classes with >5% prediction share', fontsize=11)
    ax.set_title('Effective Predicted Classes  —  C=20, final epoch', fontsize=12)
    ax.set_xticks(KS)
    ax.set_ylim(0, C + 2)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    _save(fig, OUT_CMP, 'C2_effective_classes.png')


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
        if xs:
            _alg_line(ax, alg, xs, ys)
    ax.set_xlabel('k', fontsize=11)
    ax.set_ylabel('Std of per-class accuracy (%)', fontsize=11)
    ax.set_title('Per-class Accuracy Imbalance  —  C=20, final epoch', fontsize=12)
    ax.set_xticks(KS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    _save(fig, OUT_CMP, 'D2_perclass_std.png')


# ── D3: Confusion matrices ────────────────────────────────────────────────────

def _confusion_matrix(true, pred):
    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    return cm


def _plot_confusion_for_k(k, class_names=None):
    fig, axes = plt.subplots(1, len(ALGS), figsize=(4.5 * len(ALGS), 4.5))
    d_last = None
    for ax, alg in zip(axes, ALGS):
        d = read_logits_final(alg, k)
        if d is not None:
            d_last = d
            cm      = _confusion_matrix(d['true'], d['pred'])
            cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(alg, fontsize=9, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True', fontsize=8)
        ax.set_xticks(range(C)); ax.set_yticks(range(C))
        if class_names:
            ax.set_xticklabels(class_names, rotation=90, fontsize=5)
            ax.set_yticklabels(class_names, fontsize=5)
        else:
            ax.set_xticklabels(range(C), fontsize=6)
            ax.set_yticklabels(range(C), fontsize=6)
    ep = d_last['epoch'] if d_last else '?'
    fig.suptitle(f'Confusion Matrix (row-norm)  —  C=20  k={k}  ep={ep}', fontsize=12)
    fig.tight_layout()
    _save(fig, OUT_CMP, f'D3_confusion_k{k}.png')


def plot_D3(class_names=None):
    print('D3  Confusion matrices')
    for k in [KS[0], KS[-1]]:   # k=5 and k=19
        _plot_confusion_for_k(k, class_names)


# ── E1: Accuracy trajectory heatmap ──────────────────────────────────────────

def plot_E1():
    print('E1  Accuracy trajectory heatmap')
    fig, axes = plt.subplots(1, len(ALGS), figsize=(4 * len(ALGS), 6), sharey=True)
    im_ref = None
    for ax, alg in zip(axes, ALGS):
        epoch_lists, overalls = [], []
        for k in KS:
            d = read_per_class(alg, k)
            if d is not None:
                epoch_lists.append(d['epochs'])
                overalls.append(list(d['overall']))
            else:
                epoch_lists.append([])
                overalls.append([])

        max_len    = max((len(o) for o in overalls), default=0)
        ref_epochs = next((e for e in epoch_lists if e),
                          list(range(10, 10 * max_len + 1, 10)))

        mat = np.full((max_len, len(KS)), np.nan)
        for i, col in enumerate(overalls):
            mat[:len(col), i] = col

        im_ref = ax.imshow(mat, aspect='auto', origin='lower',
                           cmap='RdYlGn', vmin=0, vmax=100, interpolation='nearest')
        ax.set_title(alg, fontsize=10, fontweight='bold')
        ax.set_xlabel('k', fontsize=9)
        ax.set_xticks(range(len(KS)))
        ax.set_xticklabels(KS, fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel('Epoch checkpoint', fontsize=9)
            ax.set_yticks(range(len(ref_epochs)))
            ax.set_yticklabels(ref_epochs, fontsize=6)

    if im_ref is not None:
        fig.colorbar(im_ref, ax=list(axes), label='Overall Accuracy (%)',
                     shrink=0.7, pad=0.02)
    fig.suptitle('Accuracy Trajectory Heatmap  —  C=20', fontsize=13)
    _save(fig, OUT_CMP, 'E1_accuracy_trajectory.png')


# ── E2: Per-class accuracy trajectory (20 figures) ───────────────────────────

def plot_E2(class_names=None):
    print('E2  Per-class accuracy trajectory (20 figures)')

    all_data   = {alg: {} for alg in ALGS}
    epochs_ref = None
    max_t      = 0
    for alg in ALGS:
        for k in KS:
            d = read_per_class(alg, k)
            if d is not None:
                all_data[alg][k] = d
                if epochs_ref is None:
                    epochs_ref = d['epochs']
                max_t = max(max_t, len(d['overall']))

    if epochs_ref is None:
        print('  [skip] no data found')
        return

    for c in range(C):
        cls_label = class_names[c] if class_names else f'class_{c}'
        fig, axes = plt.subplots(1, len(ALGS), figsize=(4 * len(ALGS), 6), sharey=True)
        fig.suptitle(
            f'Per-class Accuracy Trajectory  —  C={C}  [{c}] {cls_label}',
            fontsize=12, fontweight='bold')

        im_ref = None
        for ax, alg in zip(axes, ALGS):
            mat = np.full((max_t, len(KS)), np.nan)
            for k_idx, k in enumerate(KS):
                if k in all_data[alg]:
                    col = all_data[alg][k]['acc_mat'][:, c]
                    mat[:len(col), k_idx] = col

            im_ref = ax.imshow(mat, aspect='auto', origin='lower',
                               cmap='RdYlGn', vmin=0, vmax=100, interpolation='nearest')
            ax.set_title(alg, fontsize=10, fontweight='bold')
            ax.set_xlabel('k', fontsize=9)
            ax.set_xticks(range(len(KS)))
            ax.set_xticklabels(KS, fontsize=8)
            if ax is axes[0]:
                ax.set_ylabel('Epoch checkpoint', fontsize=9)
                ax.set_yticks(range(len(epochs_ref)))
                ax.set_yticklabels(epochs_ref, fontsize=6)

        if im_ref is not None:
            fig.colorbar(im_ref, ax=list(axes),
                         label=f'Accuracy (%)  [{cls_label}]',
                         shrink=0.7, pad=0.02)

        _save(fig, OUT_CMP, f'E2_class{c:02d}_{cls_label}.png')


# ── Main ───────────────────────────────────────────────────────────────────────

def _get_class_names(data_dir, seed):
    try:
        from src.cifar100_subset import select_cifar100_classes
        from torchvision.datasets import CIFAR100
        idxs = select_cifar100_classes(C, seed=seed)
        ds   = CIFAR100(root=data_dir, train=True, download=True)
        return [ds.classes[i] for i in idxs]
    except Exception as e:
        print(f'[warn] Could not load class names: {e}')
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_class_names', action='store_true')
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--only',     nargs='+', default=None,
                        help='Plots to generate, e.g. --only A1 C1 E1 per_run')
    args = parser.parse_args()

    class_names = _get_class_names(args.data_dir, args.seed) if args.show_class_names else None

    only = set(args.only) if args.only else None

    def run(tag, fn, *a, **kw):
        if only and tag not in only:
            return
        fn(*a, **kw)

    run('per_run', plot_per_run, class_names)
    run('A1',      plot_A1)
    run('A2',      plot_A2)
    run('B1',      plot_B1)
    run('B2',      plot_B2)
    run('C1',      plot_C1)
    run('C2',      plot_C2)
    run('D2',      plot_D2)
    run('D3',      plot_D3, class_names)
    run('E1',      plot_E1)
    run('E2',      plot_E2, class_names)

    print(f'\nDone.')
    print(f'  Per-run plots → {OUT_RUN}/')
    print(f'  Summary plots → {OUT_CMP}/')


if __name__ == '__main__':
    main()
