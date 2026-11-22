"""Algorithm runners, split by family (2026-11-22; was one 2118-line runners.py).

    simple  single-encoder losses, PRODEN, PiCO-CLS, SoLar, named PiCO / ComCo variants
    sweeps  parametrised sweep factories (biased init, alpha, conf-EMA, A/B)
    bandit  switch / bandit runners
    swap    cls-loss swap + candidate pruning
Every runner: runner(loaders, pl_ds, orig_targets, C, hparams, raw_cfg, batch_size, epochs, device,
tag, report_every) -> final accuracy. Shared epoch loop and PiCO/ComCo variants: ../families.py.
"""

from ._base import *  # noqa: F401,F403
from ._base import (_IndexedDataset, _biased_oracle_init_conf, _candidate_masked_init_conf,  # noqa: F401
                    _comco_args, _fmt_eta, _pico_args, _print_eta, _save_alpha_trace, _uniform_all_init_conf,
                    _write_trace)
from .bandit import *  # noqa: F401,F403
from .bandit import (_generic_bandit_setup, _make_generic_bandit_runner, _run_comco_weighted_bandit,  # noqa: F401
                     _run_generic_bandit)
from .simple import *  # noqa: F401,F403
from .simple import (_evaluate_argmin, _run_comco_weighted_switch_variant, _run_comco_weighted_variant,  # noqa: F401
                     _run_pico_fixed_variant, _run_pico_weighted_variant, _run_proden_variant,
                     _train_simple_shape)
from .sweeps import *  # noqa: F401,F403
from .swap import *  # noqa: F401,F403
from .swap import _run_swap_prune, _run_swap_variant  # noqa: F401
