"""
One-time download of CIFAR-10 and CIFAR-100 to the data directory.
Run this before any training script to avoid downloading mid-experiment.

    python scripts/download_data.py --data_dir data/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    args = parser.parse_args()

    from torchvision.datasets import CIFAR10, CIFAR100

    print(f"Downloading to: {os.path.abspath(args.data_dir)}\n")

    print("--- CIFAR-10 ---")
    CIFAR10(root=args.data_dir, train=True,  download=True)
    CIFAR10(root=args.data_dir, train=False, download=True)
    print("CIFAR-10 done.\n")

    print("--- CIFAR-100 ---")
    CIFAR100(root=args.data_dir, train=True,  download=True)
    CIFAR100(root=args.data_dir, train=False, download=True)
    print("CIFAR-100 done.\n")

    print("All datasets ready.")


if __name__ == '__main__':
    main()
