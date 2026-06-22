"""
One-time dataset download utility.

Download CIFAR-10 only:
    python scripts/download_data.py --cifar10 --data_dir data/

Download CIFAR-100 only:
    python scripts/download_data.py --cifar100 --data_dir data/

Download both:
    python scripts/download_data.py --cifar10 --cifar100 --data_dir data/
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--cifar10',  action='store_true', help='Download CIFAR-10')
    parser.add_argument('--cifar100', action='store_true', help='Download CIFAR-100')
    args = parser.parse_args()

    if not args.cifar10 and not args.cifar100:
        parser.error('Specify at least one of --cifar10 or --cifar100')

    from torchvision.datasets import CIFAR10, CIFAR100

    print(f"Downloading to: {os.path.abspath(args.data_dir)}\n")

    if args.cifar10:
        print("--- CIFAR-10 ---")
        CIFAR10(root=args.data_dir, train=True,  download=True)
        CIFAR10(root=args.data_dir, train=False, download=True)
        print("CIFAR-10 done.\n")

    if args.cifar100:
        print("--- CIFAR-100 ---")
        CIFAR100(root=args.data_dir, train=True,  download=True)
        CIFAR100(root=args.data_dir, train=False, download=True)
        print("CIFAR-100 done.\n")

    print("Done.")


if __name__ == '__main__':
    main()
