"""Algorithm registry: name -> AlgorithmSpec(paradigm, runner).

Every runner has the same signature so runner.py can call any of them
uniformly:

    runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size,
           epochs, device, tag, report_every) -> final_accuracy: float

To add a new algorithm: implement (or reuse) a runner function in
runners.py, add its default hyperparameters to hparams.ALGO_HPARAMS, then
add one line below. Nothing else in the pipeline needs to change.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass
class AlgorithmSpec:
    name: str
    paradigm: str  # 'PLL' or 'CLL'
    runner: Callable


def _build_registry() -> dict:
    from . import runners as r

    return {
        'CLPL':     AlgorithmSpec('CLPL',     'PLL', r.run_clpl),
        'Wu2022':   AlgorithmSpec('Wu2022',   'PLL', r.run_wu),
        'PRODEN':   AlgorithmSpec('PRODEN',   'PLL', r.run_proden),
        'PiCO':     AlgorithmSpec('PiCO',     'PLL', r.run_pico),
        'PiCO-Fixed': AlgorithmSpec('PiCO-Fixed', 'PLL', r.run_pico_fixed),
        'PiCO-MCL': AlgorithmSpec('PiCO-MCL', 'PLL', r.run_pico_mcl),
        'PiCO-SC':  AlgorithmSpec('PiCO-SC',  'PLL', r.run_pico_sc),
        'PiCO-CLS': AlgorithmSpec('PiCO-CLS', 'PLL', r.run_pico_cls),
        'SoLar':    AlgorithmSpec('SoLar',    'PLL', r.run_solar),
        'MCL-LOG':  AlgorithmSpec('MCL-LOG',  'CLL', r.run_mcl_log),
        'MCL-LOG-Fixed': AlgorithmSpec('MCL-LOG-Fixed', 'CLL', r.run_mcl_log_fixed),
        'SCL-NL':   AlgorithmSpec('SCL-NL',   'CLL', r.run_scl_nl),
        'OP':       AlgorithmSpec('OP',       'CLL', r.run_op),
        'OP-W':     AlgorithmSpec('OP-W',     'CLL', r.run_op_w),
        'CPE':      AlgorithmSpec('CPE',      'CLL', r.run_cpe),
        'ComCo':    AlgorithmSpec('ComCo',    'CLL', r.run_comco),
        'ComCo-Fixed': AlgorithmSpec('ComCo-Fixed', 'CLL', r.run_comco_fixed),
    }


ALGORITHMS = _build_registry()
ALL_ALGORITHM_NAMES = list(ALGORITHMS.keys())
