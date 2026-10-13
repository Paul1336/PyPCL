#!/usr/bin/env python
"""Paper-faithful gradient bias-variance experiment, reproducing Chou et al.
("Unbiased Risk Estimators Can Mislead: A Case Study of Learning with
Complementary Labels", ICML 2020) Section 4 protocol as closely as this
repo's CIFAR-100-subset stack allows:

  - ONE reference model per (C, seed), trained ONLY via ordinary cross-
    entropy on the TRUE label (fixed-lr SGD, no momentum/schedule) --
    complementary losses never touch the optimizer, only the diagnostic.
  - Single complementary label per sample (uniform, != true label), not
    this repo's usual k-sized candidate-set complement.
  - At every --log_every epochs (checkpoints), on the SAME fixed eval batch
    every time, resample the complementary label --m_resamples times and
    compute each loss-under-test's gradient wrt the (frozen, cached) logits,
    vs. the true-label CE oracle gradient -> Bias^2/Variance/MSE/cos_sim.

See src/pipeline/biasvariance_cll.py for the diagnostic core and
docs/scl_nl_explanation.md for the paper-alignment notes this design is
based on.

Losses compared (see the run's docstring/README discussion for rationale):
  MCL-LOG        -- Feng et al. 2020 URE, AS SHIPPED in this repo (known
                    mis-scaled unbiased-risk-estimator constant, see
                    docs/mcl_explanation.md) -- a "naive URE" data point.
  MCL-LOG-Fixed  -- same URE with the paper-correct 2(C-1)/m scaling --
                    the proper unbiased-risk-estimator reference point.
  SCL-NL         -- Chou et al. 2020's own proposed biased/low-variance
                    surrogate -- the paper's headline comparison target.
  OP             -- Liu et al. AISTATS 2023 order-preserving loss (biased,
                    non-negative risk, different family).
  OP-W           -- weighted OP variant.
  CPE-I          -- Lin & Lin PAKDD 2023 probability-estimation reduction.

Usage:
    python scripts/run_biasvariance_faithful.py --run_name biasvar_faithful_0921 \\
        --c_values 5 20 --seeds 42 43 44 --epochs 300 --log_every 20
"""

import argparse
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from src.models import create_model_for_spec
from src.pipeline.data import load_experiment_data
from src.pipeline.biasvariance import build_fixed_eval_batch
from src.pipeline.biasvariance_cll import log_checkpoint

from src.mcl_losses import MCL_LOG
from src.fixed_mcl_losses import FixedMCLLog
from src.scl_loss import SCL_NL
from src.op_loss import OPLoss, OPWLoss
from src.cpe_loss import CPELoss


def build_loss_fns(C: int):
    return {
        'MCL-LOG': MCL_LOG(num_classes=C),
        'MCL-LOG-Fixed': FixedMCLLog(num_classes=C),
        'SCL-NL': SCL_NL(),
        'OP': OPLoss(),
        'OP-W': OPWLoss(),
        'CPE-I': CPELoss(),
    }


def run_cell(C: int, seed: int, args, device, out_dir: str):
    tag = f'C={C} seed={seed}'
    loaders, pl_ds, orig_targets = load_experiment_data(
        C=C, k=1, data_dir=args.data_dir, seed=seed, log_dir=args.log_dir,
        batch_size=args.batch_size, dataset='cifar100-subset')

    model = create_model_for_spec(None, C).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)  # fixed lr, no momentum -- paper's Sec 4 setting

    eval_images, eval_true_labels, _eval_index = build_fixed_eval_batch(
        pl_ds, orig_targets, args.eval_size, device)

    loss_fns = build_loss_fns(C)
    biasvar_path = os.path.join(out_dir, 'biasvariance_faithful.csv')
    acc_path = os.path.join(out_dir, 'accuracy.csv')

    def checkpoint(epoch: int):
        model.eval()
        with torch.no_grad():
            logits = model(eval_images)
        log_checkpoint(loss_fns, logits, eval_true_labels, 'cifar100-subset', C, epoch, seed,
                        args.m_resamples, biasvar_path, device)
        model.train()

    checkpoint(0)
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for imgs, cand in loaders['pl']:
            imgs = imgs.to(device)
            true_labels = cand[:, 0].long().to(device)  # k=1 -> sole candidate IS the true label
            opt.zero_grad()
            loss = F.cross_entropy(model(imgs), true_labels)
            loss.backward()
            opt.step()

        if epoch % args.log_every == 0 or epoch == args.epochs:
            checkpoint(epoch)
            elapsed = time.perf_counter() - t0
            print(f'[{tag}] epoch {epoch}/{args.epochs}  ({elapsed:.0f}s elapsed)', flush=True)

    correct = total = 0
    model.eval()
    with torch.no_grad():
        for imgs, labels in loaders['test']:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    final_acc = 100.0 * correct / total
    os.makedirs(out_dir, exist_ok=True)
    new_file = not os.path.isfile(acc_path)
    with open(acc_path, 'a') as f:
        if new_file:
            f.write('dataset,C,seed,epochs,final_accuracy\n')
        f.write(f'cifar100-subset,{C},{seed},{args.epochs},{final_acc}\n')
    print(f'[{tag}] done. reference-model true-label test accuracy: {final_acc:.2f}%', flush=True)

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run_name', required=True)
    p.add_argument('--c_values', nargs='+', type=int, default=[5, 20])
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--epochs', type=int, default=300, help="paper's Section 4 gradient-analysis setting")
    p.add_argument('--lr', type=float, default=1e-2, help="paper's Section 4 fixed SGD lr")
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--m_resamples', type=int, default=50)
    p.add_argument('--eval_size', type=int, default=256)
    p.add_argument('--data_dir', default='./data')
    p.add_argument('--log_dir', default='logs/cifar100_subset')
    p.add_argument('--only_c', type=int, default=None)
    p.add_argument('--only_seed', type=int, default=None)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = os.path.join('results', args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'run_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    c_values = [args.only_c] if args.only_c is not None else args.c_values
    seeds = [args.only_seed] if args.only_seed is not None else args.seeds

    for C in c_values:
        for seed in seeds:
            run_cell(C, seed, args, device, out_dir)


if __name__ == '__main__':
    main()
