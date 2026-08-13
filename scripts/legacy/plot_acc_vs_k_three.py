"""
Test Accuracy vs k  —  PiCO / PiCO-Uniform / ComCo  (C=20, 500 epochs).

Output: plots/extended_summary/acc_vs_k_three.png

Usage:
    python scripts/plot_acc_vs_k_three.py
"""

import csv
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE   = 'results/extended_analysis'
OUT    = 'plots/extended_summary/acc_vs_k_three.png'
C      = 20
KS     = [5, 10, 15, 19]
EPOCHS = 500

STYLES = {
    'PiCO':         dict(color='#9467bd', marker='s', ls='-',  label='PiCO'),
    'PiCO-Uniform': dict(color='#ff7f0e', marker='D', ls='--', label='PiCO-Uniform'),
    'ComCo':        dict(color='#8c564b', marker='^', ls='-',  label='ComCo'),
}


def final_acc(alg, k):
    path = os.path.join(BASE, alg, f'C{C}_k{k}', 'per_class_loss.csv')
    if not os.path.isfile(path):
        return None
    last = None
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            last = row
    return float(last['overall_acc']) if last else None


def main():
    fig, ax = plt.subplots(figsize=(8, 6))

    for alg, st in STYLES.items():
        xs, ys = [], []
        for k in KS:
            acc = final_acc(alg, k)
            if acc is None:
                print(f'  [skip] {alg}  k={k}')
                continue
            xs.append(k)
            ys.append(acc)
        if xs:
            ax.plot(xs, ys,
                    color=st['color'], marker=st['marker'],
                    linestyle=st['ls'], linewidth=2, markersize=9,
                    label=st['label'])

    ax.set_xlabel('k  (# partial labels per sample)', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title(
        f'C = {C}  —  Test Accuracy vs k (partial labels)\n'
        f'PiCO / PiCO-Uniform / ComCo  (epoch {EPOCHS})',
        fontsize=12,
    )
    ax.set_xticks(KS)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=11)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {OUT}')


if __name__ == '__main__':
    main()
