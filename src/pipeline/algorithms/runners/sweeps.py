"""Parametrised sweeps (biased-init W / Wf / N, PiCO / ComCo alpha-weighted, confidence-EMA, A/B).
Each factory binds kwargs onto a family runner and generates the registry names."""

from ._base import *  # noqa: F401,F403
from .simple import _run_comco_weighted_variant, _run_pico_fixed_variant, _run_pico_weighted_variant, _run_proden_variant  # noqa: F401


# ─── Parametrized biased-init sweep: true_weight x {candidates, all classes} ─
#
# Generalizes run_pico_fixed_biased_init / run_proden_biased_init's fixed
# 20%/80%-one-random-candidate scheme into a sweep over (a) how much weight
# the true class gets and (b) whether the remainder is spread across the
# sample's own candidate set (biased_candidates_init) or across every class
# (biased_all_init) -- see src/pll_init.py. Registers one runner per
# (base_algorithm, strategy, true_weight) combination under a factory rather
# than writing 20 near-identical functions by hand; algorithms/__init__.py
# and hparams.py merge BIASED_SWEEP_RUNNERS into the normal registry.


def _make_pico_fixed_biased_runner(strategy: str, true_weight: float):
    """strategy: 'cand' -> biased_candidates_init (remainder spread across
    the sample's own candidates only) | 'all' -> biased_all_init (remainder
    spread across every class)."""
    from src.pll_init import biased_all_init, biased_candidates_init, biased_variant_name
    algorithm = biased_variant_name('PiCO-Fixed', strategy, true_weight)
    init_fn = biased_candidates_init if strategy == 'cand' else biased_all_init

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        init_conf = init_fn(pl_ds.targets, orig_targets, C, true_weight).to(device)
        return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                        device, tag, report_every, algorithm, init_conf)

    return algorithm, _runner


def _make_proden_biased_runner(strategy: str, true_weight: float):
    from src.pll_init import biased_variant_name
    algorithm = biased_variant_name('PRODEN', strategy, true_weight)
    init_mode = 'biased_candidates' if strategy == 'cand' else 'biased_all'

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        return _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                    device, tag, report_every, algorithm, init_mode=init_mode,
                                    true_weight=true_weight)

    return algorithm, _runner


def _build_biased_sweep_runners() -> dict:
    from src.pll_init import BIAS_WEIGHTS
    runners = {}
    for strategy in ('cand', 'all'):
        for w in BIAS_WEIGHTS:
            name, fn = _make_pico_fixed_biased_runner(strategy, w)
            runners[name] = fn
            name, fn = _make_proden_biased_runner(strategy, w)
            runners[name] = fn
    return runners


BIASED_SWEEP_RUNNERS = _build_biased_sweep_runners()


def _make_pico_weighted_runner(alpha: float):
    from src.pll_init import pico_weighted_variant_name
    algorithm = pico_weighted_variant_name(alpha)

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        return _run_pico_weighted_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                           device, tag, report_every, algorithm, alpha)

    return algorithm, _runner


def _build_pico_weighted_sweep_runners() -> dict:
    from src.pll_init import ALPHA_VALUES
    runners = {}
    for a in ALPHA_VALUES:
        name, fn = _make_pico_weighted_runner(a)
        runners[name] = fn
    return runners


PICO_WEIGHTED_SWEEP_RUNNERS = _build_pico_weighted_sweep_runners()


def _make_comco_weighted_runner(alpha: float):
    from src.pll_init import comco_weighted_variant_name
    algorithm = comco_weighted_variant_name(alpha)

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        return _run_comco_weighted_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                            device, tag, report_every, algorithm, alpha)

    return algorithm, _runner


def _build_comco_weighted_sweep_runners() -> dict:
    from src.pll_init import ALPHA_VALUES
    runners = {}
    for a in ALPHA_VALUES:
        name, fn = _make_comco_weighted_runner(a)
        runners[name] = fn
    return runners


COMCO_WEIGHTED_SWEEP_RUNNERS = _build_comco_weighted_sweep_runners()


# ─── Parametrized biased-init sweep #2: true_weight x wf (how many other ──
#     candidates the remainder is spread across, chosen at random)
#
# Interpolates between biased_oracle_init (wf=1) and biased_candidates_init
# (wf = every other candidate) -- see biased_partial_random_init in
# src/pll_init.py. Same factory pattern as BIASED_SWEEP_RUNNERS above.


def _make_pico_fixed_biased_rand_runner(true_weight: float, wf: int):
    from src.pll_init import biased_partial_random_init, biased_rand_variant_name
    algorithm = biased_rand_variant_name('PiCO-Fixed', true_weight, wf)

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        init_conf = biased_partial_random_init(pl_ds.targets, orig_targets, C, true_weight, wf).to(device)
        return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                        device, tag, report_every, algorithm, init_conf)

    return algorithm, _runner


def _make_proden_biased_rand_runner(true_weight: float, wf: int):
    from src.pll_init import biased_rand_variant_name
    algorithm = biased_rand_variant_name('PRODEN', true_weight, wf)

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        return _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                    device, tag, report_every, algorithm, init_mode='biased_partial_random',
                                    true_weight=true_weight, wf=wf)

    return algorithm, _runner


def _build_biased_rand_sweep_runners() -> dict:
    from src.pll_init import BIAS_RAND_WEIGHTS, BIAS_RAND_WF_VALUES
    runners = {}
    for w in BIAS_RAND_WEIGHTS:
        for wf in BIAS_RAND_WF_VALUES:
            name, fn = _make_pico_fixed_biased_rand_runner(w, wf)
            runners[name] = fn
            name, fn = _make_proden_biased_rand_runner(w, wf)
            runners[name] = fn
    return runners


BIASED_RAND_SWEEP_RUNNERS = _build_biased_rand_sweep_runners()


# ─── Parametrized biased-init sweep #3: true_weight x n, random pool drawn ─
#     from ALL other classes (not just the sample's own candidate set)
#
# Originally PRODEN-only (requested for a PRODEN true_weight=20% sweep over
# n=4/9/14/19 at C=20 k=5, where k-1=4 other candidates was too small a
# pool for n>4); the PiCO-Fixed counterpart was added 2026-09-15 so the
# conf_ema sweep below can run the same n-sweep on both algorithms -- see
# src/pll_init.py.biased_random_all_init / BIAS_RAND_ALL_WEIGHTS /
# BIAS_RAND_ALL_N_VALUES. Same factory pattern as BIASED_RAND_SWEEP_RUNNERS.


def _make_pico_fixed_biased_rand_all_runner(true_weight: float, n: int):
    from src.pll_init import biased_rand_all_variant_name, biased_random_all_init
    algorithm = biased_rand_all_variant_name('PiCO-Fixed', true_weight, n)

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        init_conf = biased_random_all_init(orig_targets, C, true_weight, n).to(device)
        return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                        device, tag, report_every, algorithm, init_conf)

    return algorithm, _runner


def _make_proden_biased_rand_all_runner(true_weight: float, n: int):
    from src.pll_init import biased_rand_all_variant_name
    algorithm = biased_rand_all_variant_name('PRODEN', true_weight, n)

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        return _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                    device, tag, report_every, algorithm, init_mode='biased_random_all',
                                    true_weight=true_weight, wf=n)

    return algorithm, _runner


def _build_biased_rand_all_sweep_runners() -> dict:
    from src.pll_init import BIAS_RAND_ALL_N_VALUES, BIAS_RAND_ALL_WEIGHTS
    runners = {}
    for w in BIAS_RAND_ALL_WEIGHTS:
        for n in BIAS_RAND_ALL_N_VALUES:
            name, fn = _make_pico_fixed_biased_rand_all_runner(w, n)
            runners[name] = fn
            name, fn = _make_proden_biased_rand_all_runner(w, n)
            runners[name] = fn
    return runners


BIASED_RAND_ALL_SWEEP_RUNNERS = _build_biased_rand_all_sweep_runners()


# ─── conf_ema_m sweep (2026-09-15): {init variant} x {PiCO-Fixed, PRODEN} ──
#     x CONF_EMA_SCALES
#
# Re-runs the TC-PLS W sweep and the TC-n-PLS n sweep on BOTH algorithms
# under five confidence-update EMA momentum levels, from PiCO's original
# schedule (scale 1.0) down to a hard overwrite every update (scale 0.0 ==
# original PRODEN) -- see src/pll_init.py CONF_EMA_SCALES / ema_variant_name
# and the orchestrator scripts/run_ema_sweep.py. The EMA scale is part of the
# algorithm name (e.g. 'PRODEN-BiasedCand-W20-EMA050') so every level is its
# own resumable results.csv row. The actual [start, end] range is computed
# at run time from config.yaml's pico.conf_ema_range so it follows config.


def _conf_ema_init_specs() -> list:
    """(name_fn(base) -> un-suffixed name, pico_init_fn(pl_ds, orig_targets, C)
    -> init_conf, proden_kwargs) for every init variant, in the same order
    as src.pll_init.conf_ema_sweep_base_names."""
    from src.pll_init import (BIAS_RAND_ALL_N_VALUES, BIAS_RAND_ALL_WEIGHTS, BIAS_WEIGHTS,
                              biased_candidates_init, biased_rand_all_variant_name,
                              biased_random_all_init, biased_variant_name, candidate_masked_init)
    specs = [(lambda base: base,
              lambda pl_ds, orig, C: candidate_masked_init(pl_ds.targets, C),
              dict(init_mode='candidate_masked'))]
    for w in BIAS_WEIGHTS:
        specs.append((lambda base, w=w: biased_variant_name(base, 'cand', w),
                      lambda pl_ds, orig, C, w=w: biased_candidates_init(pl_ds.targets, orig, C, w),
                      dict(init_mode='biased_candidates', true_weight=w)))
    for w in BIAS_RAND_ALL_WEIGHTS:
        for n in BIAS_RAND_ALL_N_VALUES:
            specs.append((lambda base, w=w, n=n: biased_rand_all_variant_name(base, w, n),
                          lambda pl_ds, orig, C, w=w, n=n: biased_random_all_init(orig, C, w, n),
                          dict(init_mode='biased_random_all', true_weight=w, wf=n)))
    return specs


def _make_conf_ema_runner(base: str, name_fn, pico_init_fn, proden_kwargs: dict, scale: float):
    from src.pll_init import ema_variant_name, scaled_conf_ema_range
    algorithm = ema_variant_name(name_fn(base), scale)

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        conf_ema_range = scaled_conf_ema_range(raw_cfg['pico']['conf_ema_range'], scale)
        if base == 'PiCO-Fixed':
            init_conf = pico_init_fn(pl_ds, orig_targets, C).to(device)
            return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                            device, tag, report_every, algorithm, init_conf,
                                            conf_ema_range=conf_ema_range)
        return _run_proden_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                    device, tag, report_every, algorithm, conf_ema_range=conf_ema_range,
                                    **proden_kwargs)

    return algorithm, _runner


def _build_conf_ema_sweep_runners() -> dict:
    from src.pll_init import CONF_EMA_SCALES, CONF_EMA_SWEEP_BASES, conf_ema_sweep_base_names
    runners = {}
    for name_fn, pico_init_fn, proden_kwargs in _conf_ema_init_specs():
        for base in CONF_EMA_SWEEP_BASES:
            for scale in CONF_EMA_SCALES:
                name, fn = _make_conf_ema_runner(base, name_fn, pico_init_fn, proden_kwargs, scale)
                runners[name] = fn
    # Guard: the name-only enumeration used by hparams.py / run_ema_sweep.py
    # must produce exactly the names registered here.
    from src.pll_init import ema_variant_name
    expected = {ema_variant_name(bn, s) for base in CONF_EMA_SWEEP_BASES
                for bn in conf_ema_sweep_base_names(base) for s in CONF_EMA_SCALES}
    assert set(runners) == expected, 'conf_ema sweep name enumeration out of sync'
    return runners


CONF_EMA_SWEEP_RUNNERS = _build_conf_ema_sweep_runners()


# ─── confidence-update-mechanism sweep (2026-09-16): Factor A x Factor B ──
# See src/pll_init.py's AB_VARIANT_SOURCE_HARD / AB_SWEEP_BASES / AB_SWEEP_SCALES
# docstring for the full rationale. Reuses _conf_ema_init_specs() (same
# biased-init variants as the conf_ema sweep) -- only the base's
# conf_source/conf_hard and the EMA scale list differ.


def _make_ab_runner(base: str, name_fn, pico_init_fn, scale: float):
    from src.pll_init import AB_VARIANT_SOURCE_HARD, ema_variant_name, scaled_conf_ema_range
    algorithm = ema_variant_name(name_fn(base), scale)
    conf_source, conf_hard = AB_VARIANT_SOURCE_HARD[base]

    def _runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device, tag, report_every):
        conf_ema_range = scaled_conf_ema_range(raw_cfg['pico']['conf_ema_range'], scale)
        init_conf = pico_init_fn(pl_ds, orig_targets, C).to(device)
        return _run_pico_fixed_variant(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs,
                                        device, tag, report_every, algorithm, init_conf,
                                        conf_ema_range=conf_ema_range, conf_source=conf_source,
                                        conf_hard=conf_hard)

    return algorithm, _runner


def _build_ab_sweep_runners() -> dict:
    from src.pll_init import AB_SWEEP_BASES, AB_SWEEP_SCALES, conf_ema_sweep_base_names, ema_variant_name
    runners = {}
    for name_fn, pico_init_fn, _proden_kwargs in _conf_ema_init_specs():
        for base in AB_SWEEP_BASES:
            for scale in AB_SWEEP_SCALES:
                name, fn = _make_ab_runner(base, name_fn, pico_init_fn, scale)
                runners[name] = fn
    expected = {ema_variant_name(bn, s) for base in AB_SWEEP_BASES
                for bn in conf_ema_sweep_base_names(base) for s in AB_SWEEP_SCALES}
    assert set(runners) == expected, 'AB sweep name enumeration out of sync'
    return runners


AB_SWEEP_RUNNERS = _build_ab_sweep_runners()
