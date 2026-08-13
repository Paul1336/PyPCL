"""Data loading for the CIFAR-100 class-subset comparison pipeline.

Thin wrapper around src.cifar100_subset — the actual dataset/loader logic is
not reimplemented here, only the C/k schedule (previously copy-pasted into
10+ scripts) and a single call to fetch everything needed for one (C, k) cell.
"""

from src.cifar100_subset import get_subset_dataloaders_full, prepare_cifar100_subset


def get_k_values(C: int) -> list:
    """k schedule shared by every comparison run: a few fixed k's plus
    C-proportional fractions, always including k = C-1."""
    fixed = [k for k in [1, 2, 3, 5] if k <= C - 1]
    prop = [max(1, round(r * C)) for r in [0.25, 0.50, 0.75]]
    return sorted(set(fixed + prop + [C - 1]))


def load_experiment_data(C: int, k: int, data_dir: str, seed: int, log_dir: str, batch_size: int):
    """Returns (loaders, pl_dataset_raw, original_targets) for one (C, k) cell.

    `loaders` contains 'pl', 'cl', 'pico', 'comco', 'test' DataLoaders.
    """
    pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
        total_classes=C, n_partial_labels=k,
        data_dir=data_dir, seed=seed, log_dir=log_dir,
    )
    loaders = get_subset_dataloaders_full(pl_ds, cl_ds, orig_targets, test_info, batch_size)
    return loaders, pl_ds, orig_targets
