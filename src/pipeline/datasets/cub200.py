"""CUB-200-2011 (Caltech-UCSD Birds), used as a real-world benchmark by
PiCO (Wang et al. 2022) and ComCo (Jiang et al. 2024). Requires the
`datasets` (HuggingFace) package.

The official vision.caltech.edu download URL is dead (confirmed by
scripts/probe_dataset_availability.py, 2026-08-14); this loader uses the
HuggingFace mirror `bentrevett/caltech-ucsd-birds-200-2011` instead, which
IS alive (also confirmed by the same probe).

Images are extracted once to real JPEG files under
<data_dir>/cub200_extracted/ so this dataset can reuse the lazy-path
machinery in generic_loaders.py (LazyImagePathDataset /
build_lazy_image_loaders_full) unchanged, rather than writing a
CUB-200-specific Dataset class.

Resolution note: this repo's ResNet18/SupConResNet backbones use a
CIFAR-tuned stem (3x3 stride-1 conv1, no/minimal downsampling) designed for
~32px inputs, not the 224px ImageNet-style inputs typically paired with a
pretrained backbone in the original papers' CUB-200 experiments. Feeding
full-resolution CUB-200 images through this from-scratch small-image stem
would be both architecturally mismatched and very slow on the shared
training loop. This loader resizes to 64x64 as a pragmatic middle ground --
documented here as a known fidelity gap versus the papers' own (larger,
often pretrained-backbone) setups, not hidden.
"""

import os

from src.pipeline.datasets.generic_loaders import build_lazy_image_loaders_full
from src.pipeline.datasets.specs import DatasetSpec

_HF_REPO = 'bentrevett/caltech-ucsd-birds-200-2011'
_IMAGE_SIZE = 64
_CACHE: dict = {}


def _extract_to_files(data_dir: str) -> dict:
    """Downloads (via HF `datasets`, cached by HF's own cache dir) then
    materializes each image as a real JPEG file under
    <data_dir>/cub200_extracted/{train,test}/, so downstream code can use
    plain file paths (LazyImagePathDataset) instead of holding a live HF
    Dataset object. One-time cost; skipped if already extracted."""
    if 'paths' in _CACHE:
        return _CACHE['paths']

    out_dir = os.path.join(data_dir, 'cub200_extracted')
    marker = os.path.join(out_dir, '_DONE')
    result = {'train': ([], []), 'test': ([], []), 'classes': None}

    if os.path.isfile(marker):
        print("CUB-200 already extracted, reusing cached JPEGs.", flush=True)
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
        result['classes'] = 200
        _CACHE['paths'] = result
        return result

    from datasets import load_dataset
    print(f"Loading CUB-200-2011 from HuggingFace mirror '{_HF_REPO}' "
          f"(one-time download, ~1.1GB)...", flush=True)
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

    result['classes'] = 200
    os.makedirs(out_dir, exist_ok=True)
    with open(marker, 'w') as f:
        f.write('done')
    _CACHE['paths'] = result
    return result


def _loader(C, k, data_dir, seed, log_dir, batch_size):
    raw = _extract_to_files(data_dir)
    spec = DATASETS_BY_NAME['cub200']
    if C != spec.fixed_num_classes:
        raise ValueError(f"'cub200' has {spec.fixed_num_classes} classes, got C={C}")
    train_paths, train_labels = raw['train']
    test_paths, test_labels = raw['test']
    return build_lazy_image_loaders_full(
        train_paths, train_labels, test_paths, test_labels, spec, k, batch_size,
        seed=seed, log_dir=log_dir)


def _build_specs() -> dict:
    return {
        'cub200': DatasetSpec(
            name='cub200', modality='image', backbone='cnn', fixed_num_classes=200,
            supports_pico_family=True, loader=_loader,
            in_channels=3, image_size=_IMAGE_SIZE,
            mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616),
            notes=(f'Real photos, {_IMAGE_SIZE}x{_IMAGE_SIZE} resized (see module docstring for '
                   f'why this deviates from the papers\' typically-larger/pretrained-backbone setup).'),
        ),
    }


DATASETS_BY_NAME = _build_specs()
