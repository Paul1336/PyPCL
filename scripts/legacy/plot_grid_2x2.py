"""
Combine C5 and C40 PLL/CLL comparison plots into a 2x2 grid.

Layout:
  [C5  PLL]  [C5  CLL]
  [C40 PLL]  [C40 CLL]

Input:
  plots/pll_comparison/C5_pll.png
  plots/cll_comparison/C5_cll.png
  plots/pll_comparison/C40_pll.png
  plots/cll_comparison/C40_cll.png

Output:
  plots/C5_C40_pll_cll_grid.png

Usage:
    python scripts/plot_grid_2x2.py
    python scripts/plot_grid_2x2.py --c1 5 --c2 40
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description='2x2 grid of PLL/CLL comparison plots')
    parser.add_argument('--c1',          type=int, default=5)
    parser.add_argument('--c2',          type=int, default=40)
    parser.add_argument('--pll_dir',     default='plots/pll_comparison/')
    parser.add_argument('--cll_dir',     default='plots/cll_comparison/')
    parser.add_argument('--out',         default=None,
                        help='Output path (default: plots/C{c1}_C{c2}_pll_cll_grid.png)')
    args = parser.parse_args()

    c1, c2 = args.c1, args.c2

    paths = {
        (c1, 'pll'): os.path.join(args.pll_dir, f'C{c1}_pll.png'),
        (c1, 'cll'): os.path.join(args.cll_dir, f'C{c1}_cll.png'),
        (c2, 'pll'): os.path.join(args.pll_dir, f'C{c2}_pll.png'),
        (c2, 'cll'): os.path.join(args.cll_dir, f'C{c2}_cll.png'),
    }

    for key, path in paths.items():
        if not os.path.isfile(path):
            print(f'[ERROR] Missing: {path}')
            sys.exit(1)

    fig, axes = plt.subplots(2, 2, figsize=(20, 10))

    for row, C in enumerate([c1, c2]):
        for col, paradigm in enumerate(['pll', 'cll']):
            ax = axes[row][col]
            img = mpimg.imread(paths[(C, paradigm)])
            ax.imshow(img)
            ax.axis('off')

    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    out_path = args.out or f'plots/C{c1}_C{c2}_pll_cll_grid.png'
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
