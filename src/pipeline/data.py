"""Data loading for the comparison pipeline.

'cifar100-subset' (the default, and the only dataset the pipeline supported
before 2026-08-14) is a thin wrapper around src.cifar100_subset -- kept
exactly as-is, zero behavior change. Every other --dataset value dispatches
to the src.pipeline.datasets registry instead (see that package's
__init__.py for the registry and specs.py for the DatasetSpec shape).
"""

from src.cifar100_subset import get_subset_dataloaders_full, prepare_cifar100_subset


def get_k_values(C: int) -> list:
    """k schedule shared by every comparison run: a few fixed k's plus
    C-proportional fractions, always including k = C-1."""
    fixed = [k for k in [1, 2, 3, 5] if k <= C - 1]
    prop = [max(1, round(r * C)) for r in [0.25, 0.50, 0.75]]
    return sorted(set(fixed + prop + [C - 1]))


def load_experiment_data(C: int, k: int, data_dir: str, seed: int, log_dir: str, batch_size: int,
                          dataset: str = 'cifar100-subset'):
    """Returns (loaders, pl_dataset_raw, original_targets) for one (C, k) cell.

    `loaders` contains 'pl', 'cl', 'test' DataLoaders always, plus 'pico'/
    'comco' when the dataset supports the PiCO/ComCo/SoLar family (RGB image
    datasets only -- see DatasetSpec.supports_pico_family).
    """
    if dataset == 'cifar100-subset':
        pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
            total_classes=C, n_partial_labels=k,
            data_dir=data_dir, seed=seed, log_dir=log_dir,
        )
        loaders = get_subset_dataloaders_full(pl_ds, cl_ds, orig_targets, test_info, batch_size)
        return loaders, pl_ds, orig_targets

    from src.pipeline.datasets import DATASETS
    if dataset not in DATASETS:
        raise ValueError(f"Unknown --dataset '{dataset}'. Available: "
                          f"cifar100-subset, {', '.join(sorted(DATASETS.keys()))}")
    spec = DATASETS[dataset]
    bs = spec.batch_size_override or batch_size
    return spec.loader(C, k, data_dir, seed, log_dir, bs)
