"""
Smoke test: Cour 2011 (PLL) vs MCL-LOG (CLL) on CIFAR-10.
Uses the existing CIFAR-10 infrastructure — no CIFAR-100 required.

Run:
    python scripts/smoke_test.py                     # default: k=2, 5 epochs
    python scripts/smoke_test.py --k 3 --epochs 10  # custom
"""

import argparse
import os
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_setup import get_dataloaders, prepare_cifar10_datasets
from src.training_pipelines import run_cour_training, run_mcl_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--epochs',   type=int, default=5)
    parser.add_argument('--k',        type=int, default=2,
                        help='Number of partial labels (2 ≤ k ≤ 9)')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    print(f"Dataset: CIFAR-10  |  k={args.k}  |  epochs={args.epochs}\n")

    with open('config.yaml') as f:
        cfg = yaml.safe_load(f)

    data_config  = {**cfg['data_generation'], 'cifar_path': args.data_dir}
    train_config = {'num_classes': 10}

    exp_args = argparse.Namespace(
        dataset='cifar10',
        type='constant',
        value=args.k,
        noise='clean',
        eta=0.0,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=cfg['training']['lr'],
        momentum=cfg['training']['momentum'],
        weight_decay=cfg['training']['weight_decay'],
    )

    print("Preparing CIFAR-10 datasets …")
    pl_ds, cl_ds, orig_targets = prepare_cifar10_datasets(exp_args, data_config, train_config)
    loaders, _ = get_dataloaders(exp_args, data_config, pl_ds, cl_ds, orig_targets)

    cour_accs = run_cour_training(exp_args, loaders, train_config, device)
    mcl_accs  = run_mcl_training(exp_args, loaders, train_config, device, loss_type='log')

    print("\n" + "=" * 45)
    print(f"  Cour 2011  final accuracy : {cour_accs[-1]:.2f}%")
    print(f"  MCL-LOG    final accuracy : {mcl_accs[-1]:.2f}%")
    print("=" * 45)
    print("Smoke test passed.")


if __name__ == '__main__':
    main()
