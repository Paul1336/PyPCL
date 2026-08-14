"""DatasetSpec: the declarative description of one dataset the pipeline can
train on. Mirrors src/pipeline/algorithms/__init__.py's AlgorithmSpec pattern.

See docs/00_paper_alignment_guide.md's dataset-support plan for the full
design rationale.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple


@dataclass
class DatasetSpec:
    name: str
    modality: str                        # 'image' | 'tabular' | 'preambiguous'
    backbone: str                        # 'cnn' | 'mlp'
    fixed_num_classes: Optional[int]     # None only for 'cifar100-subset' (C swept via --c_values)
    supports_pico_family: bool           # gates PiCO*/ComCo*/SoLar (need real RGB image augmentation)
    loader: Callable                     # (C, k, data_dir, seed, log_dir, batch_size) -> (loaders, pl_ds, orig_targets)

    # image-modality fields (ignored for tabular/preambiguous)
    in_channels: int = 3
    image_size: int = 32
    mean: Tuple[float, ...] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, ...] = (0.247, 0.2435, 0.2616)

    # tabular/preambiguous-modality fields
    input_dim: Optional[int] = None
    batch_size_override: Optional[int] = None

    # preambiguous-only
    is_preambiguous: bool = False

    # False for datasets whose candidate-label generation isn't k-sized (e.g.
    # cifar100-h's paper-faithful q-based generation, or is_preambiguous
    # datasets where k is meaningless) -- runner.py uses a single placeholder
    # k=1 sweep cell instead of the normal k-schedule. is_preambiguous=True
    # implies this too (checked separately in runner.py for clarity).
    sweeps_k: bool = True

    notes: str = ''


class DatasetUnavailableError(RuntimeError):
    """Raised by a DatasetSpec.loader when the underlying data source could
    not be obtained (dead link, missing local file, etc). Never silently
    substitute synthetic data -- raise this instead so the failure is
    obvious, and point at docs/dataset_availability_report.md."""
    pass
