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
    'PRODEN-RandomCandInit': _SGD,
    'MCL-LOG':  _ADAM,
    'MCL-LOG-Fixed': _ADAM,
    'SCL-NL':   _ADAM,
    'OP':       _ADAM,
    'OP-W':     _ADAM,
    'CPE':      _ADAM,
    'PiCO':     _ADAM,
    'PiCO-Oracle': _ADAM,
    'PiCO-Oracle-Add': _ADAM,
    'PiCO-Fixed': _ADAM,
    'PiCO-Fixed-UniformInit': _ADAM,
    'PiCO-Fixed-BiasedInit':  _ADAM,
    'PiCO-MOCO': _ADAM,
    'PiCO-MCL': _ADAM,
    'PiCO-MCL-Fixed': _ADAM,
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

# Parametrized PiCO-weighted-cls-loss sweep (see src/pll_init.py.ALPHA_VALUES
# / pico_weighted_variant_name and
# src/pipeline/algorithms/runners.py's PICO_WEIGHTED_SWEEP_RUNNERS).
from src.pll_init import ALPHA_VALUES, pico_weighted_variant_name  # noqa: E402

for _a in ALPHA_VALUES:
    ALGO_HPARAMS[pico_weighted_variant_name(_a)] = _ADAM

# Parametrized biased-init sweep #2 (see src/pll_init.py.BIAS_RAND_WEIGHTS /
# BIAS_RAND_WF_VALUES / biased_rand_variant_name and
# src/pipeline/algorithms/runners.py's BIASED_RAND_SWEEP_RUNNERS).
from src.pll_init import BIAS_RAND_WEIGHTS, BIAS_RAND_WF_VALUES, biased_rand_variant_name  # noqa: E402

for _w in BIAS_RAND_WEIGHTS:
    for _wf in BIAS_RAND_WF_VALUES:
        ALGO_HPARAMS[biased_rand_variant_name('PiCO-Fixed', _w, _wf)] = _ADAM
        ALGO_HPARAMS[biased_rand_variant_name('PRODEN', _w, _wf)] = _SGD

# Parametrized biased-init sweep #3 (see src/pll_init.py.BIAS_RAND_ALL_WEIGHTS
# / BIAS_RAND_ALL_N_VALUES / biased_rand_all_variant_name and
# src/pipeline/algorithms/runners.py's BIASED_RAND_ALL_SWEEP_RUNNERS).
from src.pll_init import BIAS_RAND_ALL_N_VALUES, BIAS_RAND_ALL_WEIGHTS, biased_rand_all_variant_name  # noqa: E402

for _w in BIAS_RAND_ALL_WEIGHTS:
    for _n in BIAS_RAND_ALL_N_VALUES:
        ALGO_HPARAMS[biased_rand_all_variant_name('PiCO-Fixed', _w, _n)] = _ADAM
        ALGO_HPARAMS[biased_rand_all_variant_name('PRODEN', _w, _n)] = _SGD

# conf_ema_m sweep (see src/pll_init.py CONF_EMA_SCALES / ema_variant_name /
# conf_ema_sweep_base_names and runners.py's CONF_EMA_SWEEP_RUNNERS) -- same
# optimizer as each variant's base algorithm.
from src.pll_init import (CONF_EMA_SCALES, CONF_EMA_SWEEP_BASES,  # noqa: E402
                           conf_ema_sweep_base_names, ema_variant_name)

for _base in CONF_EMA_SWEEP_BASES:
    for _bn in conf_ema_sweep_base_names(_base):
        for _s in CONF_EMA_SCALES:
            ALGO_HPARAMS[ema_variant_name(_bn, _s)] = _ADAM if _base == 'PiCO-Fixed' else _SGD

# confidence-update-mechanism sweep (Factor A x Factor B, see
# src/pll_init.py AB_VARIANT_SOURCE_HARD / AB_SWEEP_BASES / AB_SWEEP_SCALES
# and src/pipeline/algorithms/runners.py's AB_SWEEP_RUNNERS) -- all bases
# are PiCO-Fixed variants, so all use _ADAM.
from src.pll_init import AB_SWEEP_BASES, AB_SWEEP_SCALES  # noqa: E402

for _base in AB_SWEEP_BASES:
    for _bn in conf_ema_sweep_base_names(_base):
        for _s in AB_SWEEP_SCALES:
            ALGO_HPARAMS[ema_variant_name(_bn, _s)] = _ADAM


def make_optimizer(model, hparams: dict):
    import torch.optim as optim

    if hparams['optimizer'] == 'adam':
        return optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=hparams['weight_decay'])
    return optim.SGD(model.parameters(), lr=hparams['lr'],
                      momentum=hparams.get('momentum', 0.9), weight_decay=hparams['weight_decay'])
