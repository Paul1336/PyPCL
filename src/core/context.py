"""Per-cell context: everything that identifies ONE (C, k, seed) training cell.

Replaces the private keys the runner used to stash in the shared yaml dict
(raw_cfg['_current_k'], '_current_seed', '_results_dir', '_dataset_spec', '_switch_diag',
'_detail'). runner.py builds one CellContext per cell; until every runner takes it
explicitly, it is also published as raw_cfg['_ctx'] and the legacy keys are kept in sync
(context.publish), so old and new code read the same values.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CellContext:
    results_dir: str
    dataset: str
    C: int
    k: int                      # results.csv key: real k, or the negative q / bimodal sentinel
    k_mean: int                 # real positive (mean) candidate-set size, used in file names
    seed: int
    dataset_spec: Any = None
    switch_diag: bool = False
    detail: dict = field(default_factory=dict)
    config_hash: str = ''

    def publish(self, raw_cfg: dict):
        """Mirror into the legacy raw_cfg keys (removed once all runners take the context)."""
        raw_cfg['_ctx'] = self
        raw_cfg['_dataset_spec'] = self.dataset_spec
        raw_cfg['_current_k'] = self.k_mean
        raw_cfg['_current_seed'] = self.seed
        raw_cfg['_results_dir'] = self.results_dir
        raw_cfg['_switch_diag'] = self.switch_diag


def get_ctx(raw_cfg: dict) -> Optional[CellContext]:
    return (raw_cfg or {}).get('_ctx')


# Settings that change what a cell trains on / how. Diagnostics flags (detail, tsne, ...) are
# deliberately excluded: turning logging on must not invalidate finished cells.
TRAINING_KEYS = ('dataset', 'epochs', 'batch_size', 'only_q', 'only_bimodal', 'batching',
                 'holdout_frac', 'cl_bias', 'noise_eta')
YAML_KEYS = ('pico', 'comco', 'solar', 'picoplus', 'hparams_override', 'eval_split')


def training_config(cfg) -> dict:
    """The resolved settings that determine a cell's result (PipelineConfig -> plain dict)."""
    d = {k: getattr(cfg, k, None) for k in TRAINING_KEYS}
    d['yaml'] = {k: cfg.raw.get(k) for k in YAML_KEYS if cfg.raw.get(k) is not None}
    return d


def config_hash(d: dict) -> str:
    """Short stable hash of a training_config dict (12 hex chars)."""
    blob = json.dumps(d, sort_keys=True, default=str, separators=(',', ':'))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
