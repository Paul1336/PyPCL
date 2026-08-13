"""
PiCO-SC comparison: trains PiCO-SC on C=20 and plots all five PiCO-family methods.

PiCO-SC = full PiCO architecture (dual encoder, MoCo queue, prototype memory, SupConLoss)
but confidence update uses the model's own cls softmax output (PiCOCLSLoss.update_confidence)
instead of prototype similarity scores (PartialLoss.confidence_update).

Reads existing results from other scripts:
  PiCO, ComCo, PiCO-MCL  →  results/adam_comparison/gpu*/results.csv
  PiCO-CLS               →  results/pico_cls/results.csv

Writes PiCO-SC results to:
  results/pico_sc/results.csv

Generates plot:
  plots/pico_sc/pico_sc_comparison.png

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_pico_sc_comparison.py
    CUDA_VISIBLE_DEVICES=0 python scripts/run_pico_sc_comparison.py --only_k 7 --epochs 5
"""

import argparse
import csv
import gc
import glob
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import get_subset_dataloaders_full, prepare_cifar100_subset
from src.engine import evaluate_model, train_pico_sc_epoch
from src.pico_cls_loss import PiCOCLSLoss
from src.pico.model import PiCOModel
from src.pico.utils_loss import SupConLoss

# ─── Constants ────────────────────────────────────────────────────────────────

C          = 20
LR         = 3e-4
BS         = 512
WD         = 1e-4
REPORT_EVERY = 10
ALG_NAME   = 'PiCO-SC'

STYLES = {
    'PiCO':     dict(color='#9467bd', marker='s', linestyle='--', linewidth=2, markersize=6),
    'ComCo':    dict(color='#8c564b', marker='^', linestyle='-',  linewidth=2, markersize=6),
    'PiCO-MCL': dict(color='#bcbd22', marker='p', linestyle=':',  linewidth=2, markersize=6),
    'PiCO-CLS': dict(color='#e377c2', marker='*', linestyle='-',  linewidth=2, markersize=7),
    'PiCO-SC':  dict(color='#98df8a', marker='h', linestyle='--', linewidth=2, markersize=6),
}
PLOT_ALGOS = ['PiCO', 'ComCo', 'PiCO-MCL', 'PiCO-CLS', 'PiCO-SC']

# ─── k schedule ───────────────────────────────────────────────────────────────

def get_k_values(c: int) -> list:
    fixed = [k for k in [1, 2, 3, 5] if k <= c - 1]
    prop  = [max(1, round(r * c)) for r in [0.25, 0.50, 0.75]]
    return sorted(set(fixed + prop + [c - 1]))

# ─── ETA helpers ──────────────────────────────────────────────────────────────

def _fmt_eta(s: float) -> str:
    if s < 90:   return f'{s:.0f}s'
    if s < 3600: return f'{s/60:.1f}min'
    return f'{s/3600:.2f}h'

def _print_eta(tag: str, ep_done: int, ep_total: int, t_chunk: float, chunk_size: int):
    avg_s = t_chunk / chunk_size
    eta   = avg_s * (ep_total - ep_done)
    print(f'  [{tag}]  ep {ep_done:>3}/{ep_total}  '
          f'{avg_s:.1f}s/ep  ETA {_fmt_eta(eta)}', flush=True)

# ─── CSV helpers ──────────────────────────────────────────────────────────────

_CSV_FIELDS = ['total_classes', 'n_partial_labels', 'algorithm',
               'final_accuracy', 'epochs', 'training_time_s', 'timestamp']


def _load_done(csv_path: str) -> set:
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            done.add((int(row['total_classes']),
                      int(row['n_partial_labels']),
                      row['algorithm']))
    return done


def _append_result(csv_path, c, k, alg, acc, epochs, elapsed_s):
    new_file = not os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            'total_classes':    c,
            'n_partial_labels': k,
            'algorithm':        alg,
            'final_accuracy':   round(acc, 4),
            'epochs':           epochs,
            'training_time_s':  round(elapsed_s, 1),
            'timestamp':        datetime.now().isoformat(),
        })


def _load_all_results(base_dirs: list) -> dict:
    """Merge CSVs from multiple source directories → res[C][alg][k] = accuracy."""
    res: dict = {}
    seen: set = set()
    for base in base_dirs:
        for pat in [os.path.join(base, 'results.csv'),
                    os.path.join(base, 'gpu*', 'results.csv')]:
            for path in sorted(glob.glob(pat)):
                if not os.path.isfile(path):
                    continue
                with open(path, newline='') as f:
                    for row in csv.DictReader(f):
                        key = (row['total_classes'], row['n_partial_labels'], row['algorithm'])
                        if key in seen:
                            continue
                        seen.add(key)
                        C_  = int(row['total_classes'])
                        k_  = int(row['n_partial_labels'])
                        alg = row['algorithm']
                        acc = float(row['final_accuracy'])
                        res.setdefault(C_, {}).setdefault(alg, {})[k_] = acc
    return res

# ─── Training ─────────────────────────────────────────────────────────────────

def _train_pico_sc(loaders: dict, c: int, pico_config: dict,
                   pl_ds, epochs: int, device, tag: str) -> float:
    """Full PiCO with softmax-based confidence update (no prototype dependency for cls)."""
    pico_args = {
        'num_class':      c,
        'epochs':         epochs,
        'low_dim':        pico_config['low_dim'],
        'moco_queue':     pico_config['moco_queue'],
        'moco_m':         pico_config['moco_m'],
        'proto_m':        pico_config['proto_m'],
        'prot_start':     pico_config['prot_start'],
        'loss_weight':    pico_config['loss_weight'],
        'conf_ema_range': pico_config['conf_ema_range'],
    }
    model     = PiCOModel(pico_args).to(device)
    cls_loss  = PiCOCLSLoss(
        pl_ds.targets, c,
        conf_ema_range=tuple(pico_config['conf_ema_range']),
        epochs=epochs,
    ).to(device)
    cont_loss = SupConLoss()
    opt       = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        cls_loss.set_conf_ema_m(ep)   # PiCOCLSLoss.set_conf_ema_m takes only epoch
        train_pico_sc_epoch(pico_args, model, loaders['pico'],
                            cls_loss, cont_loss, opt, ep, device)

        if (ep + 1) % REPORT_EVERY == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(REPORT_EVERY, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    acc = evaluate_model(model, loaders['test'], device)
    del model, cls_loss, cont_loss, opt
    gc.collect()
    torch.cuda.empty_cache()
    return acc

# ─── Plotting ─────────────────────────────────────────────────────────────────

def _draw(ax, alg: str, k_acc: dict):
    if not k_acc:
        return
    ks, accs = zip(*sorted(k_acc.items()))
    ax.plot(ks, accs, label=alg, **STYLES[alg])


def _setup_ax(ax, title: str, y_max: int, ylabel: bool = False):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=9)
    if ylabel:
        ax.set_ylabel('Test Accuracy (%)', fontsize=9)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)


def make_plot(res: dict, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    vals = [acc for c_d in res.values()
            for alg_d in c_d.values()
            for acc in alg_d.values()]
    ym = 80 if not vals else int(np.ceil(max(vals) / 10) * 10)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.suptitle('PiCO-SC vs PiCO-CLS vs PiCO-MCL vs PiCO vs ComCo  (C=20)', fontsize=12)
    _setup_ax(ax, f'C = {C}', ym, ylabel=True)
    for alg in PLOT_ALGOS:
        _draw(ax, alg, res.get(C, {}).get(alg, {}))
    ax.legend(fontsize=9, loc='best')
    fig.tight_layout()

    path = os.path.join(plots_dir, 'pico_sc_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [plot] → {path}', flush=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train PiCO-SC (C=20) and compare with PiCO, ComCo, PiCO-MCL, PiCO-CLS.')
    parser.add_argument('--data_dir',    default='./data')
    parser.add_argument('--out_dir',     default='results/pico_sc/')
    parser.add_argument('--adam_dir',    default='results/adam_comparison/',
                        help='Directory with run_adam_comparison CSVs (PiCO, ComCo, PiCO-MCL)')
    parser.add_argument('--pico_cls_dir', default='results/pico_cls/',
                        help='Directory with run_pico_cls_comparison CSVs (PiCO-CLS)')
    parser.add_argument('--plots_dir',   default='plots/pico_sc/')
    parser.add_argument('--log_dir',     default='logs/cifar100_subset')
    parser.add_argument('--config',      default='config.yaml')
    parser.add_argument('--epochs',      type=int, default=200)
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--only_k',      type=int, default=None,
                        help='Run only this k value (e.g. 7). Default: all k values.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pico_config = cfg['pico']

    os.makedirs(args.out_dir,   exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, 'results.csv')
    done     = _load_done(csv_path)

    k_vals = get_k_values(C)
    if args.only_k is not None:
        k_vals = [args.only_k]

    print(f'Device={device}  C={C}  epochs={args.epochs}  lr={LR}  bs={BS}')
    print(f'k schedule: {k_vals}')
    print(f'Resume: {len(done)} entries in {csv_path}\n', flush=True)

    for k in k_vals:
        if (C, k, ALG_NAME) in done:
            print(f'  [skip] k={k}', flush=True)
            # Still regenerate plot in case other sources have new data
            res = _load_all_results([args.out_dir, args.adam_dir, args.pico_cls_dir])
            make_plot(res, args.plots_dir)
            continue

        print(f'\n--- C={C}  k={k} ---', flush=True)
        pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
            total_classes=C, n_partial_labels=k,
            data_dir=args.data_dir, seed=args.seed, log_dir=args.log_dir,
        )
        loaders = get_subset_dataloaders_full(pl_ds, cl_ds, orig_targets, test_info, BS)

        tag = f'{ALG_NAME} C={C} k={k}'
        t0  = time.perf_counter()
        acc = _train_pico_sc(loaders, C, pico_config, pl_ds, args.epochs, device, tag)
        elapsed = time.perf_counter() - t0

        _append_result(csv_path, C, k, ALG_NAME, acc, args.epochs, elapsed)
        print(f'  [done] k={k}  acc={acc:.2f}%  ({elapsed/60:.1f} min)', flush=True)

        res = _load_all_results([args.out_dir, args.adam_dir, args.pico_cls_dir])
        make_plot(res, args.plots_dir)

    print('\nAll done.')


if __name__ == '__main__':
    main()
