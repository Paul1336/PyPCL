"""SUN397 (Scene UNderstanding), used as a supplementary fine-grained
benchmark by ComCo (Jiang et al. 2024, Section 5.6) alongside CUB-200 -- see
docs/comco_explanation.md's benchmark section.

STATUS (2026-08-14): code-complete, following the exact same pattern as
cub200.py (HF `datasets` -> extract to real JPEG files -> lazy-path loading
via build_lazy_image_loaders_full), but NOT download-verified in this pass.
Unlike every other dataset in this registry, this one was a deliberate scope
cut: the official torchvision.datasets.SUN397 source is ~37GB (108,754
images), and even the smallest HF mirror found (`tanganke/sun397`) is still
~17GB across its train+test splits -- both far larger than every other
dataset handled this session, and this is only ComCo's supplementary
Section 5.6 experiment, not one of its primary benchmarks (unlike CUB-200,
which is Section 5.6 for ComCo but a MAIN benchmark for PiCO, and was fully
downloaded+verified). Written so a future session/the user's server (more
bandwidth/disk budget) can run it as-is without writing new code -- but
treat the first real run as unverified until it actually completes once.

If this turns out to still be too large in practice, `torchvision.datasets.SUN397`
(confirmed to exist, see scripts/probe_dataset_availability.py) is the
fallback path -- not used here only because the `datasets` library's caching
made CUB-200 simpler to extract to lazy file paths uniformly.
"""

import os

from src.pipeline.datasets.generic_loaders import build_lazy_image_loaders_full
from src.pipeline.datasets.specs import DatasetSpec

_HF_REPO = 'tanganke/sun397'
_IMAGE_SIZE = 64
_N_CLASSES = 397
_CACHE: dict = {}


def _extract_to_files(data_dir: str) -> dict:
    """Same pattern as cub200.py's _extract_to_files -- see that module for
    the reasoning. Not exercised end-to-end in this pass (see module
    docstring): the download itself (~17GB via this HF mirror) was not
    triggered."""
    if 'paths' in _CACHE:
        return _CACHE['paths']

    out_dir = os.path.join(data_dir, 'sun397_extracted')
    marker = os.path.join(out_dir, '_DONE')
    result = {'train': ([], []), 'test': ([], [])}

    if os.path.isfile(marker):
        print("SUN397 already extracted, reusing cached JPEGs.", flush=True)
        for split in ('train', 'test'):
            split_dir = os.path.join(out_dir, split)
            paths, labels = [], []
            for cls_name in sorted(os.listdir(split_dir)):
                cls_dir = os.path.join(split_dir, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    paths.append(os.path.join(cls_dir, fname))
                    labels.append(int(cls_name))
            result[split] = (paths, labels)
        _CACHE['paths'] = result
        return result

    from datasets import load_dataset
    print(f"Loading SUN397 from HuggingFace mirror '{_HF_REPO}' "
          f"(one-time download, ~17GB -- this WILL take a while)...", flush=True)
    hf_ds = load_dataset(_HF_REPO)

    for split in ('train', 'test'):
        paths, labels = [], []
        split_dir = os.path.join(out_dir, split)
        for i, ex in enumerate(hf_ds[split]):
            label = int(ex['label'])
            cls_dir = os.path.join(split_dir, f'{label:03d}')
            os.makedirs(cls_dir, exist_ok=True)
            fpath = os.path.join(cls_dir, f'{i}.jpg')
            if not os.path.isfile(fpath):
                ex['image'].convert('RGB').save(fpath, format='JPEG', quality=90)
            paths.append(fpath)
            labels.append(label)
        result[split] = (paths, labels)
        print(f"  extracted {len(paths)} '{split}' images to {split_dir}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    with open(marker, 'w') as f:
        f.write('done')
    _CACHE['paths'] = result
    return result


def _loader(C, k, data_dir, seed, log_dir, batch_size):
    raw = _extract_to_files(data_dir)
    spec = DATASETS_BY_NAME['sun397']
    if C != spec.fixed_num_classes:
        raise ValueError(f"'sun397' has {spec.fixed_num_classes} classes, got C={C}")
    train_paths, train_labels = raw['train']
    test_paths, test_labels = raw['test']
    return build_lazy_image_loaders_full(
        train_paths, train_labels, test_paths, test_labels, spec, k, batch_size,
        seed=seed, log_dir=log_dir)


def _build_specs() -> dict:
    return {
        'sun397': DatasetSpec(
            name='sun397', modality='image', backbone='cnn', fixed_num_classes=_N_CLASSES,
            supports_pico_family=True, loader=_loader,
            in_channels=3, image_size=_IMAGE_SIZE,
            mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616),
            notes=('UNVERIFIED (2026-08-14): code-complete but the ~17GB download was not run in '
                   'this pass -- see module docstring. Run a small smoke test before trusting results.'),
        ),
    }


DATASETS_BY_NAME = _build_specs()
