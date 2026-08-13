"""
List the selected classes for a given C and seed.

Usage:
    python scripts/list_classes.py --C 20
    python scripts/list_classes.py --C 10 --seed 1
"""
import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cifar100_subset import select_cifar100_classes
from torchvision.datasets import CIFAR100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--C',        type=int, default=20)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--data_dir', default='./data')
    args = parser.parse_args()

    dataset = CIFAR100(root=args.data_dir, train=True, download=True)
    selected_indices = select_cifar100_classes(args.C, seed=args.seed)
    selected_names   = [dataset.classes[i] for i in selected_indices]

    print(f'C={args.C}  seed={args.seed}')
    print(f'{"idx":>4}  {"CIFAR-100 idx":>13}  class')
    print('-' * 35)
    for local_idx, (cifar_idx, name) in enumerate(zip(selected_indices, selected_names)):
        print(f'{local_idx:>4}  {cifar_idx:>13}  {name}')

if __name__ == '__main__':
    main()
