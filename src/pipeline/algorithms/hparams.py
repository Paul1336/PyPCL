"""Per-algorithm default hyperparameters.

Optimizer/lr/wd choices mirror the values already validated in
scripts/run_adam_comparison.py, scripts/run_solar_comparison.py, etc.:
Adam lr=3e-4 for most algorithms; SGD lr=0.01 momentum=0.9 for PRODEN and
SoLar, which follow their original papers.
"""

_ADAM = dict(optimizer='adam', lr=3e-4, weight_decay=1e-4)
_SGD = dict(optimizer='sgd', lr=0.01, momentum=0.9, weight_decay=1e-4)

ALGO_HPARAMS = {
    'CLPL':     _ADAM,
    'Wu2022':   _ADAM,
    'PRODEN':   _SGD,
    'PRODEN-UniformInit': _SGD,
    'PRODEN-BiasedInit':  _SGD,
    'MCL-LOG':  _ADAM,
    'MCL-LOG-Fixed': _ADAM,
    'SCL-NL':   _ADAM,
    'OP':       _ADAM,
    'OP-W':     _ADAM,
    'CPE':      _ADAM,
    'PiCO':     _ADAM,
    'PiCO-Oracle': _ADAM,
    'PiCO-Fixed': _ADAM,
    'PiCO-Fixed-UniformInit': _ADAM,
    'PiCO-Fixed-BiasedInit':  _ADAM,
    'PiCO-MOCO': _ADAM,
    'PiCO-MCL': _ADAM,
    'PiCO-SC':  _ADAM,
    'PiCO-CLS': _ADAM,
    'ComCo':    _ADAM,
    'ComCo-Fixed': _ADAM,
    'SoLar':    _SGD,
}

# Parametrized biased-init sweep (see src/pll_init.py.BIAS_WEIGHTS /
# biased_variant_name and src/pipeline/algorithms/runners.py's
# BIASED_SWEEP_RUNNERS) -- same optimizer as each variant's base algorithm.
from src.pll_init import BIAS_WEIGHTS, biased_variant_name  # noqa: E402

for _w in BIAS_WEIGHTS:
    for _strategy in ('cand', 'all'):
        ALGO_HPARAMS[biased_variant_name('PiCO-Fixed', _strategy, _w)] = _ADAM
        ALGO_HPARAMS[biased_variant_name('PRODEN', _strategy, _w)] = _SGD


def make_optimizer(model, hparams: dict):
    import torch.optim as optim

    if hparams['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    return optim.SGD(model.parameters(), lr=hparams['lr'],
                      momentum=hparams.get('momentum', 0.9), weight_decay=hparams['weight_decay'])
