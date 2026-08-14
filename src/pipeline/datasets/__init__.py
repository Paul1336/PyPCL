"""Dataset registry: name -> DatasetSpec. Mirrors
src/pipeline/algorithms/__init__.py's ALGORITHMS registry pattern.

'cifar100-subset' is NOT in this registry -- it's the original, default
dataset and is handled as a special case directly in src/pipeline/data.py
(zero behavior change to the already-validated 6-paper experiment pipeline).
Every entry here is an *additional* dataset on top of that default.

To add a new dataset: implement a loader module (see image_builtin.py /
uci_tabular.py / cub200.py / pll_mat.py for the patterns), add its
DatasetSpec entries to _build_registry() below. Nothing else needs to change.
"""

from src.pipeline.datasets.specs import DatasetSpec, DatasetUnavailableError


def _build_registry() -> dict:
    from src.pipeline.datasets import image_builtin

    registry = {}
    registry.update(image_builtin.DATASETS_BY_NAME)

    # Phase 2 (tabular/text) -- added only if their optional dependencies
    # (ucimlrepo, scikit-learn) are importable, so a bare pipeline install
    # doesn't hard-require them just to run --dataset cifar100-subset.
    try:
        from src.pipeline.datasets import uci_tabular
        registry.update(uci_tabular.DATASETS_BY_NAME)
    except ImportError:
        pass
    try:
        from src.pipeline.datasets import text_20news
        registry.update(text_20news.DATASETS_BY_NAME)
    except ImportError:
        pass

    # Phase 3 (CUB-200, CIFAR-100-H, SUN397).
    try:
        from src.pipeline.datasets import cub200
        registry.update(cub200.DATASETS_BY_NAME)
    except ImportError:
        pass
    from src.pipeline.datasets import cifar100_h  # no optional deps, always available
    registry.update(cifar100_h.DATASETS_BY_NAME)
    try:
        from src.pipeline.datasets import sun397
        registry.update(sun397.DATASETS_BY_NAME)
    except ImportError:
        pass

    # Phase 4 (real-world PLL .mat data + CLPL original data) -- these
    # register even if their extraction dependencies are missing; the
    # DatasetSpec.loader itself raises DatasetUnavailableError with a clear
    # message when actually invoked, rather than disappearing from --dataset's
    # choices list.
    try:
        from src.pipeline.datasets import pll_mat
        registry.update(pll_mat.DATASETS_BY_NAME)
    except ImportError:
        pass
    try:
        from src.pipeline.datasets import clpl_tv
        registry.update(clpl_tv.DATASETS_BY_NAME)
    except ImportError:
        pass

    return registry


DATASETS = _build_registry()
ALL_DATASET_NAMES = ['cifar100-subset'] + list(DATASETS.keys())
