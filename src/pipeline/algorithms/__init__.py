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

    registry = {
        'CLPL':     AlgorithmSpec('CLPL',     'PLL', r.run_clpl),
        'Wu2022':   AlgorithmSpec('Wu2022',   'PLL', r.run_wu),
        'PRODEN':   AlgorithmSpec('PRODEN',   'PLL', r.run_proden),
        'PRODEN-UniformInit': AlgorithmSpec('PRODEN-UniformInit', 'PLL', r.run_proden_uniform_init),
        'PRODEN-BiasedInit':  AlgorithmSpec('PRODEN-BiasedInit',  'PLL', r.run_proden_biased_init),
        'PRODEN-RandomCandInit': AlgorithmSpec('PRODEN-RandomCandInit', 'PLL', r.run_proden_random_cand_init),
        'PiCO':     AlgorithmSpec('PiCO',     'PLL', r.run_pico),
        'PiCO-Oracle': AlgorithmSpec('PiCO-Oracle', 'PLL', r.run_pico_oracle),
        'PiCO-Oracle-Add': AlgorithmSpec('PiCO-Oracle-Add', 'PLL', r.run_pico_oracle_add),
        'PiCO-Fixed': AlgorithmSpec('PiCO-Fixed', 'PLL', r.run_pico_fixed),
        'PiCO-Fixed-UniformInit': AlgorithmSpec('PiCO-Fixed-UniformInit', 'PLL', r.run_pico_fixed_uniform_init),
        'PiCO-Fixed-BiasedInit':  AlgorithmSpec('PiCO-Fixed-BiasedInit',  'PLL', r.run_pico_fixed_biased_init),
        'PiCO-MOCO': AlgorithmSpec('PiCO-MOCO', 'PLL', r.run_pico_moco),
        'PiCO-MCL': AlgorithmSpec('PiCO-MCL', 'PLL', r.run_pico_mcl),
        'PiCO-MCL-Fixed': AlgorithmSpec('PiCO-MCL-Fixed', 'PLL', r.run_pico_mcl_fixed),
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

    # Parametrized biased-init sweep (PiCO-Fixed-Biased{Cand,All}-W* /
    # PRODEN-Biased{Cand,All}-W*, both PLL) -- see runners.BIASED_SWEEP_RUNNERS.
    for name, fn in r.BIASED_SWEEP_RUNNERS.items():
        registry[name] = AlgorithmSpec(name, 'PLL', fn)

    # Parametrized PiCO-weighted-cls-loss sweep (alpha * PiCO-Fixed's
    # PartialLoss + (1-alpha) * PiCOMCLLoss, PLL) -- see
    # runners.PICO_WEIGHTED_SWEEP_RUNNERS and src/pll_init.py's ALPHA_VALUES.
    for name, fn in r.PICO_WEIGHTED_SWEEP_RUNNERS.items():
        registry[name] = AlgorithmSpec(name, 'PLL', fn)

    # ComCo-side mirror (alpha * ComCo-Fixed's own cls loss + (1-alpha) *
    # PartialLoss borrowed onto ComCo's architecture, CLL) -- see
    # runners.COMCO_WEIGHTED_SWEEP_RUNNERS and src/pll_init.py's ALPHA_VALUES.
    for name, fn in r.COMCO_WEIGHTED_SWEEP_RUNNERS.items():
        registry[name] = AlgorithmSpec(name, 'CLL', fn)

    # Per-batch alpha-switching variants (variable-average-k experiment) --
    # see runners.run_comco_weighted_batchcond / run_comco_weighted_batchrandom.
    registry['ComCo-Weighted-BatchCond'] = AlgorithmSpec('ComCo-Weighted-BatchCond', 'CLL',
                                                          r.run_comco_weighted_batchcond)
    registry['ComCo-Weighted-BatchRandom'] = AlgorithmSpec('ComCo-Weighted-BatchRandom', 'CLL',
                                                            r.run_comco_weighted_batchrandom)
    registry['ComCo-Weighted-SampleCond'] = AlgorithmSpec('ComCo-Weighted-SampleCond', 'CLL',
                                                           r.run_comco_weighted_samplecond)
    registry['ComCo-Weighted-BanditPaired'] = AlgorithmSpec('ComCo-Weighted-BanditPaired', 'CLL',
                                                             r.run_comco_weighted_bandit_paired)
    registry['ComCo-Weighted-BanditEpoch'] = AlgorithmSpec('ComCo-Weighted-BanditEpoch', 'CLL',
                                                            r.run_comco_weighted_bandit_epoch)

    # cls-loss swap experiment ({PiCO-Fixed, ComCo-Fixed} x every cls loss,
    # 'SwapPiCO-<loss>' / 'SwapComCo-<loss>') -- see runners.SWAP_RUNNERS and
    # src/clsloss_swap.py. Paradigm follows the framework.
    for name, fn in r.SWAP_RUNNERS.items():
        registry[name] = AlgorithmSpec(name, 'PLL' if name.startswith('SwapPiCO') else 'CLL', fn)

    # Parametrized biased-init sweep #2 (PiCO-Fixed-BiasedRand-W*-Wf* /
    # PRODEN-BiasedRand-W*-Wf*, both PLL) -- see runners.BIASED_RAND_SWEEP_RUNNERS.
    for name, fn in r.BIASED_RAND_SWEEP_RUNNERS.items():
        registry[name] = AlgorithmSpec(name, 'PLL', fn)

    # Parametrized biased-init sweep #3 ({PiCO-Fixed,PRODEN}-BiasedRandAll-W*-N*,
    # random pool drawn from ALL classes rather than just the candidate set,
    # PLL) -- see runners.BIASED_RAND_ALL_SWEEP_RUNNERS.
    for name, fn in r.BIASED_RAND_ALL_SWEEP_RUNNERS.items():
        registry[name] = AlgorithmSpec(name, 'PLL', fn)

    # conf_ema_m sweep (every init variant above x {PiCO-Fixed, PRODEN} x
    # five EMA momentum levels, '...-EMA100' .. '...-EMA000', PLL) -- see
    # runners.CONF_EMA_SWEEP_RUNNERS and scripts/run_ema_sweep.py.
    for name, fn in r.CONF_EMA_SWEEP_RUNNERS.items():
        registry[name] = AlgorithmSpec(name, 'PLL', fn)

    # confidence-update-mechanism sweep (Factor A: prototype- vs
    # classifier-sourced confidence update, Factor B: hard one-hot vs soft
    # distribution -- every init variant above x {SrcSoftmax, SoftUpdate,
    # SrcSoftmax-SoftUpdate} x {EMA100, EMA050, EMA000}, PLL) -- see
    # runners.AB_SWEEP_RUNNERS and scripts/run_ab_sweep.py.
    for name, fn in r.AB_SWEEP_RUNNERS.items():
        registry[name] = AlgorithmSpec(name, 'PLL', fn)

    return registry


ALGORITHMS = _build_registry()
ALL_ALGORITHM_NAMES = list(ALGORITHMS.keys())
