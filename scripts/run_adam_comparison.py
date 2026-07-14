"""
All-8-methods sweep: C=5 and C=40.
Optimizer: Adam  lr=3e-4  bs=512  epochs=200.

8-GPU parallel split (one algorithm per GPU):
  GPU 0 → Cour2011   GPU 1 → Wu2022   GPU 2 → PRODEN    GPU 3 → MCL-LOG
  GPU 4 → SCL-NL     GPU 5 → PiCO     GPU 6 → PiCO-MCL  GPU 7 → ComCo

Each GPU writes to its own CSV (results/adam_comparison/gpu{id}/results.csv).
Plots are updated from all gpu*/results.csv after each (C, k) completes.
ETA is printed every 10 epochs.

Commands:
  CUDA_VISIBLE_DEVICES=0 python scripts/run_adam_comparison.py --gpu_id 0
  CUDA_VISIBLE_DEVICES=1 python scripts/run_adam_comparison.py --gpu_id 1
  ...
  CUDA_VISIBLE_DEVICES=7 python scripts/run_adam_comparison.py --gpu_id 7

Single-GPU smoke test:
  python scripts/run_adam_comparison.py --gpu_id 0 --num_gpus 1 --epochs 5
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
from src.clpl_loss import CLPLSquaredHingeLoss
from src.comco.model import ComCoModel
from src.comco.utils_loss import ComCoCLSLoss, ComCoContrastiveLoss
from src.engine import (evaluate_model, train_algorithm,
                         train_comco_epoch, train_pico_epoch,
                         train_pico_mclloss_epoch)
from src.mcl_losses import MCL_LOG
from src.models import create_model
from src.pico.mcl_cls_loss import PiCOMCLLoss
from src.pico.model import PiCOModel
from src.pico.utils_loss import PartialLoss, SupConLoss
from src.proden_loss import proden
from src.scl_loss import SCL_NL
from src.wu_loss import WuPLLLoss

# ─── Constants ────────────────────────────────────────────────────────────────

C_VALUES     = [5, 20]
LR           = 3e-4
BS           = 512
WD           = 1e-4
REPORT_EVERY = 10      # print ETA every N epochs

PLL_ALGOS = ['Cour2011', 'PRODEN', 'PiCO', 'PiCO-MCL']
CLL_ALGOS = ['MCL-LOG', 'SCL-NL', 'ComCo']
ALL_ALGOS = PLL_ALGOS + CLL_ALGOS   # index 0-6 → GPU 1-7

# ─── Visual style (shared across all 4 figures) ───────────────────────────────

STYLES = {
    'Cour2011': dict(color='#1f77b4', marker='o', linestyle='-',  linewidth=2, markersize=6),
    'Wu2022':   dict(color='#17becf', marker='D', linestyle='--', linewidth=2, markersize=6),
    'PRODEN':   dict(color='#2ca02c', marker='^', linestyle='-',  linewidth=2, markersize=6),
    'PiCO':     dict(color='#9467bd', marker='s', linestyle='--', linewidth=2, markersize=6),
    'PiCO-MCL': dict(color='#bcbd22', marker='p', linestyle=':',  linewidth=2, markersize=6),
    'MCL-LOG':  dict(color='#d62728', marker='o', linestyle='-',  linewidth=2, markersize=6),
    'SCL-NL':   dict(color='#ff7f0e', marker='D', linestyle='--', linewidth=2, markersize=6),
    'ComCo':    dict(color='#8c564b', marker='^', linestyle='-',  linewidth=2, markersize=6),
}

# ─── k-value schedule ─────────────────────────────────────────────────────────

def get_k_values(C: int) -> list:
    fixed = [k for k in [1, 2, 3, 5] if k <= C - 1]
    prop  = [max(1, round(r * C)) for r in [0.25, 0.50, 0.75]]
    return sorted(set(fixed + prop + [C - 1]))

# ─── ETA helper ───────────────────────────────────────────────────────────────

def _fmt_eta(seconds: float) -> str:
    if seconds < 90:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds/60:.1f}min'
    return f'{seconds/3600:.2f}h'


def _print_eta(tag: str, ep_done: int, ep_total: int,
               t_chunk: float, chunk_size: int):
    """Print progress and ETA after each report chunk."""
    avg_s = t_chunk / chunk_size
    remaining = ep_total - ep_done
    eta = avg_s * remaining
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


def _load_all_results(base_dir: str) -> dict:
    """Merge all gpu*/results.csv into res[C][alg][k] = accuracy."""
    res: dict = {}
    patterns = [
        os.path.join(base_dir, 'results.csv'),
        os.path.join(base_dir, 'gpu*', 'results.csv'),
    ]
    seen: set = set()
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            if not os.path.isfile(path):
                continue
            with open(path, newline='') as f:
                for row in csv.DictReader(f):
                    key = (row['total_classes'], row['n_partial_labels'], row['algorithm'])
                    if key in seen:
                        continue
                    seen.add(key)
                    C   = int(row['total_classes'])
                    k   = int(row['n_partial_labels'])
                    alg = row['algorithm']
                    acc = float(row['final_accuracy'])
                    res.setdefault(C, {}).setdefault(alg, {})[k] = acc
    return res


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

# ─── Training helpers ─────────────────────────────────────────────────────────

def _adam(params):
    return optim.Adam(params, lr=LR, weight_decay=WD)


def _train_simple_eta(loss_fn, loader_key: str, loaders: dict,
                      C: int, epochs: int, device, tag: str) -> float:
    """Run simple method in REPORT_EVERY-epoch chunks, printing ETA each chunk."""
    model = create_model(C).to(device)
    opt   = _adam(model.parameters())
    last_accs = []

    for ep_start in range(0, epochs, REPORT_EVERY):
        chunk = min(REPORT_EVERY, epochs - ep_start)
        t0 = time.perf_counter()
        last_accs = train_algorithm(model, loaders[loader_key], loaders['test'],
                                    loss_fn, opt, chunk, device)
        elapsed = time.perf_counter() - t0
        _print_eta(tag, ep_start + chunk, epochs, elapsed, chunk)

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return last_accs[-1]


def _train_pico_eta(loaders: dict, C: int, pico_config: dict,
                    pl_ds, epochs: int, device, tag: str) -> float:
    pico_args = {
        'num_class':      C,
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
    init_conf = torch.ones(len(pl_ds), C).to(device) / C
    cls_loss  = PartialLoss(init_conf)
    cont_loss = SupConLoss()
    opt       = _adam(model.parameters())

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        cls_loss.set_conf_ema_m(ep, pico_args)
        train_pico_epoch(pico_args, model, loaders['pico'],
                         cls_loss, cont_loss, opt, ep, device)

        if (ep + 1) % REPORT_EVERY == 0 or ep + 1 == epochs:
            elapsed = time.perf_counter() - chunk_t0
            _print_eta(tag, ep + 1, epochs, elapsed, min(REPORT_EVERY, ep + 1))
            chunk_t0 = time.perf_counter()
            gc.collect()
            torch.cuda.empty_cache()

    acc = evaluate_model(model, loaders['test'], device)
    del model, cls_loss, cont_loss, opt, init_conf
    gc.collect()
    torch.cuda.empty_cache()
    return acc


def _train_pico_mcl_eta(loaders: dict, C: int, pico_config: dict,
                         epochs: int, device, tag: str) -> float:
    pico_args = {
        'num_class':      C,
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
    cls_loss  = PiCOMCLLoss()
    cont_loss = SupConLoss()
    opt       = _adam(model.parameters())

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        train_pico_mclloss_epoch(pico_args, model, loaders['pico'],
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


def _train_comco_eta(loaders: dict, C: int, comco_config: dict,
                     epochs: int, device, tag: str) -> float:
    comco_args = {
        'num_class':   C,
        'epochs':      epochs,
        'low_dim':     comco_config['low_dim'],
        'moco_queue':  comco_config['moco_queue'],
        'moco_m':      comco_config['moco_m'],
        'loss_weight': comco_config['loss_weight'],
        'temperature': comco_config['temperature'],
        'top_k':       comco_config['top_k'],
        'warmup_neg':  comco_config['warmup_neg'],
        'warmup_pos':  comco_config['warmup_pos'],
    }
    model     = ComCoModel(comco_args).to(device)
    cls_loss  = ComCoCLSLoss()
    cont_loss = ComCoContrastiveLoss(temperature=comco_args['temperature'],
                                     top_k=comco_args['top_k'])
    opt = _adam(model.parameters())

    chunk_t0 = time.perf_counter()
    for ep in range(epochs):
        train_comco_epoch(comco_args, model, loaders['comco'],
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

# ─── Plot helpers ─────────────────────────────────────────────────────────────

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
    ax.set_xlabel('k  (# partial labels per sample)', fontsize=9)
    if ylabel:
        ax.set_ylabel('Test Accuracy (%)', fontsize=9)
    ax.set_ylim(0, y_max)
    ax.grid(True, alpha=0.3)


def make_plots(res: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    ym = _global_ymax(res)

    # Figure 1 — 2×2 PLL/CLL × C5/C20
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('PLL vs CLL  —  C=5 and C=20', fontsize=13)
    for row, C in enumerate(C_VALUES):
        for col, (algos, paradigm) in enumerate([(PLL_ALGOS, 'PLL'), (CLL_ALGOS, 'CLL')]):
            ax = axes[row][col]
            _setup_ax(ax, f'{paradigm}  —  C = {C}', ym, ylabel=(col == 0))
            for alg in algos:
                _draw(ax, alg, res.get(C, {}).get(alg, {}))
            ax.legend(fontsize=8, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig1_pll_cll.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Figure 2 — PiCO / ComCo / PiCO-MCL
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('PiCO vs ComCo vs PiCO-MCL', fontsize=13)
    for col, C in enumerate(C_VALUES):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in ['PiCO', 'ComCo', 'PiCO-MCL']:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig2_pico_comco_picomcl.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Figure 3 — ComCo / MCL-LOG
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('ComCo vs MCL-LOG', fontsize=13)
    for col, C in enumerate(C_VALUES):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in ['ComCo', 'MCL-LOG']:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig3_comco_mcl.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Figure 4 — Cour2011 / PRODEN / SCL-NL / ComCo
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle('Cour2011 vs PRODEN vs SCL-NL vs ComCo', fontsize=13)
    for col, C in enumerate(C_VALUES):
        ax = axes[col]
        _setup_ax(ax, f'C = {C}', ym, ylabel=(col == 0))
        for alg in ['Cour2011', 'PRODEN', 'SCL-NL', 'ComCo']:
            _draw(ax, alg, res.get(C, {}).get(alg, {}))
        ax.legend(fontsize=9, loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fig4_cour_proden_scl_comco.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f'  [plots] → {out_dir}', flush=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='All-8-methods sweep on C=5, C=40. Adam lr=3e-4 bs=512.')
    parser.add_argument('--gpu_id',    type=int, default=0,
                        help='GPU index (0-based). Determines which algorithm(s) to run.')
    parser.add_argument('--num_gpus',  type=int, default=7,
                        help='Total number of parallel GPU workers.')
    parser.add_argument('--data_dir',  default='./data')
    parser.add_argument('--out_dir',   default='results/adam_comparison/')
    parser.add_argument('--plots_dir', default='plots/adam_comparison/')
    parser.add_argument('--log_dir',   default='logs/cifar100_subset')
    parser.add_argument('--config',    default='config.yaml')
    parser.add_argument('--epochs',    type=int, default=200)
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pico_config  = cfg['pico']
    comco_config = cfg['comco']

    # Algorithm assignment: round-robin by gpu_id
    my_algos = [alg for i, alg in enumerate(ALL_ALGOS)
                if i % args.num_gpus == args.gpu_id]

    # Per-GPU CSV path to avoid concurrent write conflicts
    gpu_dir  = os.path.join(args.out_dir, f'gpu{args.gpu_id}')
    csv_path = os.path.join(gpu_dir, 'results.csv')
    os.makedirs(gpu_dir,        exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    done = _load_done(csv_path)

    print(f'GPU {args.gpu_id}/{args.num_gpus}  device={device}  '
          f'epochs={args.epochs}  lr={LR}  bs={BS}', flush=True)
    print(f'Algorithms: {my_algos}', flush=True)
    print(f'k schedule: C=5→{get_k_values(5)}  C=40→{get_k_values(40)}', flush=True)
    print(f'Resume: {len(done)} entries already in {csv_path}\n', flush=True)

    for C in C_VALUES:
        k_vals = get_k_values(C)
        total_k = len(k_vals)
        print(f'\n{"="*65}', flush=True)
        print(f'C = {C}   k = {k_vals}', flush=True)
        print(f'{"="*65}', flush=True)

        for k_idx, k in enumerate(k_vals):
            pending = [a for a in my_algos if (C, k, a) not in done]
            if not pending:
                print(f'  [skip] C={C} k={k}', flush=True)
                continue

            print(f'\n--- C={C}  k={k}  ({k_idx+1}/{total_k})  pending: {pending} ---',
                  flush=True)

            pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
                total_classes=C, n_partial_labels=k,
                data_dir=args.data_dir, seed=args.seed, log_dir=args.log_dir,
            )
            loaders = get_subset_dataloaders_full(
                pl_ds, cl_ds, orig_targets, test_info, BS
            )

            for alg in my_algos:
                if (C, k, alg) in done:
                    continue
                tag = f'GPU{args.gpu_id} {alg} C={C} k={k}'
                print(f'\n  >> {alg}  C={C}  k={k}', flush=True)
                t0 = time.perf_counter()

                if alg == 'Cour2011':
                    acc = _train_simple_eta(
                        CLPLSquaredHingeLoss(), 'pl',
                        loaders, C, args.epochs, device, tag)
                elif alg == 'Wu2022':
                    acc = _train_simple_eta(
                        WuPLLLoss(), 'pl',
                        loaders, C, args.epochs, device, tag)
                elif alg == 'PRODEN':
                    acc = _train_simple_eta(
                        proden(), 'pl',
                        loaders, C, args.epochs, device, tag)
                elif alg == 'MCL-LOG':
                    acc = _train_simple_eta(
                        MCL_LOG(num_classes=C), 'cl',
                        loaders, C, args.epochs, device, tag)
                elif alg == 'SCL-NL':
                    acc = _train_simple_eta(
                        SCL_NL(), 'cl',
                        loaders, C, args.epochs, device, tag)
                elif alg == 'PiCO':
                    acc = _train_pico_eta(
                        loaders, C, pico_config,
                        pl_ds, args.epochs, device, tag)
                elif alg == 'PiCO-MCL':
                    acc = _train_pico_mcl_eta(
                        loaders, C, pico_config,
                        args.epochs, device, tag)
                elif alg == 'ComCo':
                    acc = _train_comco_eta(
                        loaders, C, comco_config,
                        args.epochs, device, tag)

                elapsed = time.perf_counter() - t0
                _append_result(csv_path, C, k, alg, acc, args.epochs, elapsed)
                done.add((C, k, alg))
                print(f'  DONE  {alg}  acc={acc:.2f}%  total={_fmt_eta(elapsed)}',
                      flush=True)

            del loaders, pl_ds, cl_ds, orig_targets, test_info
            gc.collect()
            torch.cuda.empty_cache()

            # Update 4 plots from all GPUs' results
            res = _load_all_results(args.out_dir)
            make_plots(res, args.plots_dir)

    print(f'\nGPU {args.gpu_id} finished.', flush=True)
    print(f'  CSV   → {csv_path}', flush=True)
    print(f'  Plots → {args.plots_dir}', flush=True)


if __name__ == '__main__':
    main()
