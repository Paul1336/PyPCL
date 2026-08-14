"""CIFAR-100-H: the "hierarchical" partial-label setting used by PiCO
(Wang et al., ICLR 2022) alongside CIFAR-10/CIFAR-100/CUB-200 -- see
docs/pico_explanation.md's benchmark section.

Not a separate dataset download -- it's the same CIFAR-100 images, with
candidate labels drawn from the SAME coarse superclass as the true label
(of CIFAR-100's standard 20 fine->coarse superclasses) instead of uniformly
from all classes, making the partial-label task harder (candidates are
visually/semantically similar to the true class).

Generation process -- CONFIRMED 2026-08-14 by extracting the actual PDF text
of the paper (pypdf, not the earlier PDF-rendering-tool-unavailable guess):
Section 4.4 states verbatim "CIFAR-100 with hierarchical labels (CIFAR-100-H),
where we generate candidate labels that belong to the same superclass. We set
q=0.5 for CIFAR-100 with hierarchical labels" (Table 6 also reports q in
{0.1, 0.5, 0.8}). This is q-based (each same-superclass false label
independently included with probability q), NOT a fixed candidate-set size k
-- see ComparisonDataGenerator.generate_pl_dataset_hierarchical_variable in
src/data_utils.py. Because this repo's pipeline is built around a k-sweep
CLI, this dataset uses DatasetSpec.sweeps_k=False and bypasses the k-schedule
entirely (single placeholder cell, same mechanism as pre-ambiguous datasets),
picking q via _HIERARCHICAL_Q (default 0.5, the paper's headline Table 3
setting) rather than a --q CLI flag, to avoid threading a new parameter
through the whole config/CLI stack for one dataset.

Reuses prepare_cifar100_subset's raw-data loading (same download, same class
-subset-selection mechanics) with hierarchical_q=<q>, so C is swept via
--c_values exactly like the default 'cifar100-subset'. Meaningful mainly at
large C (close to 100): with a small class subset, most superclasses won't
have their other members selected, so most samples fall back to uniform
sampling (logged at runtime, not silent) -- confirmed by a real run at C=20
(85% fallback rate).
"""

from src.cifar100_subset import get_subset_dataloaders_full, prepare_cifar100_subset
from src.pipeline.datasets.specs import DatasetSpec

_HIERARCHICAL_Q = 0.5  # paper's Table 3 headline setting (Table 6 also tests 0.1, 0.8)


def _loader(C, k, data_dir, seed, log_dir, batch_size):
    # k is ignored (see module docstring) -- this dataset is q-based, not
    # k-based, and DatasetSpec.sweeps_k=False means runner.py only ever
    # passes a k=1 placeholder here anyway.
    pl_ds, cl_ds, orig_targets, test_info, _ = prepare_cifar100_subset(
        total_classes=C, n_partial_labels=1, data_dir=data_dir, seed=seed,
        log_dir=log_dir, hierarchical_q=_HIERARCHICAL_Q)
    loaders = get_subset_dataloaders_full(pl_ds, cl_ds, orig_targets, test_info, batch_size)
    return loaders, pl_ds, orig_targets


def _build_specs() -> dict:
    return {
        'cifar100-h': DatasetSpec(
            name='cifar100-h', modality='image', backbone='cnn', fixed_num_classes=None,
            supports_pico_family=True, loader=_loader, sweeps_k=False,
            in_channels=3, image_size=32, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616),
            notes=(f'Hierarchical (same-superclass) candidate labels on CIFAR-100, q={_HIERARCHICAL_Q} '
                   f'(paper-confirmed generation process, see module docstring). Most meaningful at '
                   f'large --c_values (close to 100); small subsets mostly fall back to uniform sampling.'),
        ),
    }


DATASETS_BY_NAME = _build_specs()
