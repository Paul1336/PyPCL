"""Per-algorithm default hyperparameters (rule-based since 2026-11-22).

Adam lr=3e-4 for every algorithm, except PRODEN* and SoLar, which use SGD lr=0.01 momentum=0.9
as in their papers (Swap*-PRODEN etc. run inside the PiCO/ComCo frameworks and use Adam).
This reproduces the old 508-entry hand-written table exactly (tests/golden/data/_registry.json),
minus its one stale entry with no runner (PCD-ComCo-OracleProto). Per-run overrides: the yaml
key `hparams_override` (see runner.py).

ALGO_HPARAMS is built lazily from the registry, so importing make_optimizer no longer pulls in
pll_init / clsloss_swap / pcdiag (that import side effect used to force lazy-import workarounds).
"""

_ADAM = dict(optimizer='adam', lr=3e-4, weight_decay=1e-4)
_SGD = dict(optimizer='sgd', lr=0.01, momentum=0.9, weight_decay=1e-4)


def hparams_for(name: str) -> dict:
    return dict(_SGD if name.startswith(('PRODEN', 'SoLar')) else _ADAM)


_CACHE = {}


def __getattr__(attr):
    if attr == 'ALGO_HPARAMS':
        if not _CACHE:
            from src.pipeline.algorithms import ALGORITHMS
            _CACHE.update({n: hparams_for(n) for n in ALGORITHMS})
        return _CACHE
    raise AttributeError(attr)


def make_optimizer(model, hparams: dict):
    import torch.optim as optim

    if hparams['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    return optim.SGD(model.parameters(), lr=hparams['lr'],
                      momentum=hparams.get('momentum', 0.9), weight_decay=hparams['weight_decay'])
