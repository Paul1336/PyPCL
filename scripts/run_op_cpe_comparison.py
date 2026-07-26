"""
OP, OP-W, and CPE comparison vs existing CLL baselines.

  OP    : Liu et al. AISTATS 2023  — L = -log softmax(-g(x))_{ȳ}
            negated-logit CE; standard argmax inference
  OP-W  : Liu et al. AISTATS 2023  — weighted version of OP (Definition 4.1)
            w(g,y) = softmax(u+1)_y · softmax(g)_y + ε, u_j = 1/softmax(-g)_j
  CPE   : Lin & Lin PAKDD 2023 (CPE-I variant)
            L = -log softmax(g(x))_{ȳ}  (CE treating CL as positive target)
            *** inference uses argmin(f(x)), NOT argmax ***

SCL-NL / MCL-LOG / ComCo results are read from adam_comparison CSVs for reference.
Own results are written to  results/op_cpe_comparison/gpu{id}/results.csv.

Figures after each completed (C, k):
  fig1_op_cpe_vs_cll.png  — OP, OP-W, CPE, SCL-NL, MCL-LOG, ComCo
  fig2_op_cpe.png         — OP, OP-W, and CPE only

Algorithm assignments with 3 algos (round-robin by gpu_id % num_gpus):
  --gpu_id 0 --num_gpus 3  →  OP
  --gpu_id 1 --num_gpus 3  →  OP-W
  --gpu_id 2 --num_gpus 3  →  CPE
  --gpu_id 0 --num_gpus 1  →  all three (default, sequential)

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_op_cpe_comparison.py
  CUDA_VISIBLE_DEVICES=0 python scripts/run_op_cpe_comparison.py --only_c 5 --only_k 3
  CUDA_VISIBLE_DEVICES=0 python scripts/run_op_cpe_comparison.py --algo OP
  CUDA_VISIBLE_DEVICES=0 python scripts/run_op_cpe_comparison.py --gpu_id 0 --num_gpus 3
  CUDA_VISIBLE_DEVICES=1 python scripts/run_op_cpe_comparison.py --gpu_id 1 --num_gpus 3
  CUDA_VISIBLE_DEVICES=2 python scripts/run_op_cpe_comparison.py --gpu_id 2 --num_gpus 3
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
from src.engine import train_algorithm, evaluate_model
from src.models import create_model
from src.op_loss import OPLoss, OPWLoss
from src.cpe_loss import CPELoss

# ─── Constants ────────────────────────────────────────────────────────────────

C_VALUES     = [5, 20]
LR           = 3e-4
BS           = 512
WD           = 1e-4
REPORT_EVERY = 10

ALL_ALGOS = ['OP', 'OP-W', 'CPE']

# ─── Visual styles ────────────────────────────────────────────────────────────

_RENAME = {'Cour2011': 'CLPL'}

STYLES = {
    'OP':      dict(color='#e377c2', marker='D', linestyle='-',  linewidth=2, markersize=6),
    'OP-W':    dict(color='#9467bd', marker='P', linestyle='-',  linewidth=2, markersize=6),
    'CPE':     dict(color='#17becf', marker='s', linestyle='-',  linewidth=2, markersize=6),
    'SCL-NL':  dict(color='#ff7f0e', marker='D', linestyle='--', linewidth=2, markersize=6),
    'MCL-LOG': dict(color='#d62728', marker='o', linestyle='-',  linewidth=2, markersize=6),
    'ComCo':   dict(color='#8c564b', marker='^', linestyle='-',  linewidth=2, markersize=6),
}

# ─── k-value schedule (same as run_adam_comparison) ───────────────────────────

def get_k_values(C: int) -> list:
    fixed = [k for k in [1, 2, 3, 5] if k <= C - 1]
    prop  = [max(1, round(r * C)) for r in [0.25, 0.50, 0.75]]
    return sorted(set(fixed + prop + [C - 1]))

# ─── ETA helper ───────────────────────────────────────────────────────────────

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


def _append_result(csv_path, C, k, alg, acc, epochs, elapsed_s):
    new_file = not os.path.isfile(csv_path)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if new_file:
            w.writeheader()
        w.writerow({
            'total_classes':    C,
            'n_partial_labels': k,
            'algorithm':        alg,
            'final_accuracy':   round(acc, 4),
            'epochs':           epochs,
            'training_time_s':  round(elapsed_s, 1),
            'timestamp':        datetime.now().isoformat(),
        })


def _load_all_results(base_dirs: list) -> dict:
    """Merge CSVs from own dir + adam_comparison → res[C][alg][k] = accuracy."""
    res  = {}
    seen = set()
    for base in base_dirs:
        for pat in [os.path.join(base, 'results.csv'),
                    os.path.join(base, 'gpu*', 'results.csv')]:
            for path in sorted(glob.glob(pat)):
                with open(path, newline='') as f:
                    for row in csv.DictReader(f):
                        key = (row['total_classes'], row['n_partial_labels'], row['algorithm'])
                        if key in seen:
                            continue
                        seen.add(key)
                        C_  = int(row['total_classes'])
                        k_  = int(row['n_partial_labels'])
                        alg = _RENAME.get(row['algorithm'], row['algorithm'])
                        acc = float(row['final_accuracy'])
                        res.setdefault(C_, {}).setdefault(alg, {})[k_] = acc
    return res

# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate_argmin(model, loader, device) -> float:
    """CPE-I inference: argmin f(x) predicts the true class.

    CPE-I trains f to predict P(ȳ|x), so the class with the LOWEST predicted
    complementary-label probability is the most likely true class.
    """
    model.eval()
    correct = total = 0
    for batch in loader:
        imgs, labels = batch[0].to(device), batch[1].to(device)
        preds   = model(imgs).argmin(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total

# ─── Training ─────────────────────────────────────────────────────────────────

def _train_simple(loss_fn, loaders: dict, C: int, epochs: int, device, tag: str) -> float:
    """Adam + standard CL loader + argmax evaluation (OP and OP-W)."""
    model = create_model(C).to(device)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    final_acc = 0.0
    for ep_start in range(0, epochs, REPORT_EVERY):
        chunk = min(REPORT_EVERY, epochs - ep_start)
        t0    = time.perf_counter()
        accs  = train_algorithm(model, loaders['cl'], loaders['test'],
                                loss_fn, opt, chunk, device)
        elapsed = time.perf_counter() - t0
        final_acc = accs[-1]
        _print_eta(tag, ep_start + chunk, epochs, elapsed, chunk)

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc


def _train_cpe(loss_fn, loaders: dict, C: int, epochs: int, device, tag: str) -> float:
    """Adam + CL training with CPE-I loss + argmin evaluation.

    CPE-I trains the model to predict P(ȳ|x) directly, so inference must use
    argmin(f(x)) to recover the true-class prediction.  We cannot reuse
    train_algorithm here because it evaluates with argmax internally.
    """
    model = create_model(C).to(device)
    opt   = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    final_acc = 0.0
    for ep_start in range(0, epochs, REPORT_EVERY):
        chunk   = min(REPORT_EVERY, epochs - ep_start)
        t0      = time.perf_counter()

        for _ in range(chunk):
            model.train()
            for batch in loaders['cl']:
                imgs      = batch[0].to(device)
                cl_labels = batch[1].to(device)
                opt.zero_grad()
                loss_fn(model(imgs), cl_labels).backward()
                opt.step()

        elapsed   = time.perf_counter() - t0
        final_acc = _evaluate_argmin(model, loaders['test'], device)
        _print_eta(tag, ep_start + chunk, epochs, elapsed, chunk)

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return final_acc

# ─── Plotting ─────────────────────────────────────────────────────────────────

def _global_ymax(res: dict) -> int:
    vals = [acc for C_d in res.values()
            for alg_d in C_d.values()
            for acc in alg_d.values()]
    return 80 if not vals else int(np.ceil(max(vals) / 10) * 10)


def _draw(ax, alg: str, k_acc: dict):
    if not k_acc:
        return
    ks, accs = zip(*sorted(k_acc.items()))
    ax.plot(ks, accs, label=alg, **STYLES[alg])


def _setup_ax(ax, title: str, y_max: int, ylabel: bool = False):
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('k  (# complementary labels per sample)', fontsize=9)
    if ylabel:
        ax.set_ylabel('Test Accuracy (%)', fontsize=9)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)


def make_plots(res: dict, plots_dir: str):
    os.makedirs(plots_dir, exist_ok=True)
    ym = _global_ymax(res)

    # Figure 1 — OP / OP-W / CPE / SCL-NL / MCL-LOG / ComCo
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('OP / OP-W / CPE vs CLL baselines  —  C=5 and C=20', fontsize=13)
    ref_algos = ['OP', 'OP-W', 'CPE', 'SCL-NL', 'MCL-LOG', 'ComCo']
    for col, C in enumerate(C_VALUES):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in ref_algos:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'fig1_op_cpe_vs_cll.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Figure 2 — OP, OP-W, and CPE only
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('OP vs OP-W vs CPE  —  C=5 and C=20', fontsize=13)
    for col, C in enumerate(C_VALUES):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in ['OP', 'OP-W', 'CPE']:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'fig2_op_cpe.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'  [plots] → {plots_dir}', flush=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id',    type=int, default=0)
    parser.add_argument('--num_gpus',  type=int, default=1)
    parser.add_argument('--data_dir',  default='./data')
    parser.add_argument('--out_dir',   default='results/op_cpe_comparison/')
    parser.add_argument('--adam_dir',  default='results/adam_comparison/',
                        help='adam_comparison result dir for reference baselines')
    parser.add_argument('--plots_dir', default='plots/op_cpe_comparison/')
    parser.add_argument('--log_dir',   default='logs/cifar100_subset')
    parser.add_argument('--epochs',    type=int, default=200)
    parser.add_argument('--seed',      type=int, default=42)
    parser.add_argument('--only_c',    type=int, default=None)
    parser.add_argument('--only_k',    type=int, default=None)
    parser.add_argument('--algo',      type=str, default=None,
                        help='Override algorithm (OP, OP-W, or CPE). Ignores gpu_id round-robin.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.algo is not None:
        my_algos = [args.algo]
    else:
        my_algos = [alg for i, alg in enumerate(ALL_ALGOS)
                    if i % args.num_gpus == args.gpu_id]

    gpu_dir  = os.path.join(args.out_dir, f'gpu{args.gpu_id}')
    csv_path = os.path.join(gpu_dir, 'results.csv')
    os.makedirs(gpu_dir,        exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    done = _load_done(csv_path)

    print(f'GPU {args.gpu_id}/{args.num_gpus}  device={device}  '
          f'epochs={args.epochs}  lr={LR}  bs={BS}', flush=True)
    print(f'Algorithms : {my_algos}', flush=True)
    print(f'k schedule : C=5→{get_k_values(5)}  C=20→{get_k_values(20)}', flush=True)
    print(f'Resume     : {len(done)} entries in {csv_path}\n', flush=True)

    c_values = [args.only_c] if args.only_c is not None else C_VALUES

    for C in c_values:
        k_vals = get_k_values(C)
        if args.only_k is not None:
            k_vals = [args.only_k]
        print(f'\n{"="*60}', flush=True)
        print(f'C = {C}   k = {k_vals}', flush=True)
        print(f'{"="*60}', flush=True)

        for k_idx, k in enumerate(k_vals):
            pending = [a for a in my_algos if (C, k, a) not in done]
            if not pending:
                print(f'  [skip] C={C} k={k}', flush=True)
                continue

            print(f'\n--- C={C}  k={k}  ({k_idx+1}/{len(k_vals)})  pending: {pending} ---',
                  flush=True)

            pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
                total_classes=C, n_partial_labels=k,
                data_dir=args.data_dir, seed=args.seed, log_dir=args.log_dir,
            )
            loaders = get_subset_dataloaders_full(pl_ds, cl_ds, orig_targets, test_info, BS)

            for alg in my_algos:
                if (C, k, alg) in done:
                    continue

                tag = f'GPU{args.gpu_id} {alg} C={C} k={k}'
                print(f'\n  >> {alg}  C={C}  k={k}', flush=True)
                t0 = time.perf_counter()

                if alg == 'OP':
                    acc = _train_simple(OPLoss(),  loaders, C, args.epochs, device, tag)
                elif alg == 'OP-W':
                    acc = _train_simple(OPWLoss(), loaders, C, args.epochs, device, tag)
                elif alg == 'CPE':
                    acc = _train_cpe(CPELoss(),    loaders, C, args.epochs, device, tag)
                else:
                    raise ValueError(f'Unknown algorithm: {alg}')

                elapsed = time.perf_counter() - t0
                _append_result(csv_path, C, k, alg, acc, args.epochs, elapsed)
                done.add((C, k, alg))
                print(f'  DONE  {alg}  acc={acc:.2f}%  total={_fmt_eta(elapsed)}', flush=True)

            del loaders, pl_ds, cl_ds, orig_targets, test_info
            gc.collect()
            torch.cuda.empty_cache()

            res = _load_all_results([args.out_dir, args.adam_dir])
            make_plots(res, args.plots_dir)

    print(f'\nGPU {args.gpu_id} finished.', flush=True)
    print(f'  CSV   → {csv_path}', flush=True)
    print(f'  Plots → {args.plots_dir}', flush=True)


if __name__ == '__main__':
    main()
