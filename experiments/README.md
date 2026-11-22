# experiments/

One declarative spec per experiment, run with `scripts/sweep.py` (replaces the per-experiment
`logs/*/launch.py` and `scripts/launch_*.py` launchers):

    python scripts/sweep.py dry-run experiments/<id>.yaml   # every cell's command (DONE marked)
    python scripts/sweep.py status  experiments/<id>.yaml
    tmux new -s <id> 'python scripts/sweep.py run experiments/<id>.yaml'
    echo pause|drain|resume > logs/<id>/control

Spec format: `src/sweep/spec.py` docstring. The grid key order is the expansion order and fixes
the shard ids (persisted in `logs/<id>/wid.json`). `noctr_1118.yaml` is a port of an old
launcher, checked cell-for-cell by `tests/sweep_equiv.py`.
