"""Experiment loop: for each (C, k, algorithm) not already recorded, load
data, train, and append the result. Ties together gpu / data / algorithms /
results — this is the only module that orchestrates all of them."""

import gc
import os
import random
import time

import numpy as np
import torch

from src.core.context import CellContext, config_hash, training_config
from src.pipeline import data, results
from src.pipeline.algorithms import ALGORITHMS
from src.pipeline.algorithms.hparams import ALGO_HPARAMS
from src.pipeline.config import PipelineConfig
from src.pipeline.gpu import assign_algorithms, get_device

# Algorithms that need real RGB weak/strong augmentation pairs (loaders['pico'] / ['comco']) --
# skipped on DatasetSpecs with supports_pico_family=False. Derived from the registry
# (AlgorithmSpec.needs_views) since 2026-11-22; the old hand list missed most newer families.
IMAGE_ONLY_ALGORITHMS = {n for n, s in ALGORITHMS.items() if s.needs_views}


def _set_seed(seed: int) -> None:
    """Seeds the global Python/NumPy/PyTorch RNGs so everything downstream of
    this call for the rest of this cell -- PL/CL candidate-label sampling
    (ComparisonDataGenerator draws from the global np.random state), model
    weight init, DataLoader shuffling, and data augmentation -- is
    reproducible per --seeds value. Deliberately does NOT touch which C
    classes get selected: select_cifar100_classes uses its own independent,
    fixed-seed np.random.RandomState (see
    src.cifar100_subset.CLASS_SELECTION_SEED), so it's unaffected by this.

    Called twice per (C, k, algorithm, seed) cell below: once before data
    generation, and again (same seed) right before each algorithm's own
    runner call. The second call matters because loaders/pl_ds are shared
    across every algorithm in `pending` for a given (C, k, seed) -- without
    re-seeding per algorithm, an algorithm's model init would depend on how
    much RNG state the *other* algorithms that happened to run before it (in
    this worker process) had already consumed, making a single (C, k,
    algorithm, seed) cell's result depend on unrelated scheduling details
    (multi-GPU round-robin sharding, --algo pinning, resumed partial runs)
    instead of being reproducible on its own."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dataset_spec(dataset_name: str):
    """Returns the DatasetSpec for `dataset_name`, or None for the original
    'cifar100-subset' path (kept as None so create_model_for_spec's
    `spec is None` branch reproduces the exact original behavior)."""
    if dataset_name == 'cifar100-subset':
        return None
    from src.pipeline.datasets import DATASETS
    return DATASETS[dataset_name]


def run(cfg: PipelineConfig):
    device = get_device()
    my_algorithms = assign_algorithms(cfg.algorithms, cfg.gpu_id, cfg.num_gpus, cfg.algo_override)

    spec = _dataset_spec(cfg.dataset)
    if cfg.only_q is not None and spec is not None and not spec.supports_q:
        raise ValueError(f"--only_q is not supported for dataset '{cfg.dataset}' "
                          f"(no q-based generation wired up for it yet); use --only_k instead.")
    if spec is not None and not spec.supports_pico_family:
        skipped = [a for a in my_algorithms if a in IMAGE_ONLY_ALGORITHMS]
        if skipped:
            print(f"[skip] {skipped} require real RGB image augmentation; "
                  f"dataset '{cfg.dataset}' does not support it ({spec.notes})", flush=True)
        my_algorithms = [a for a in my_algorithms if a not in IMAGE_ONLY_ALGORITHMS]

    shard = results.shard_path(cfg.results_dir, cfg.gpu_id)
    done_hash = results.load_done_hashes(shard)
    done = set(done_hash)
    train_cfg = training_config(cfg)
    chash = config_hash(train_cfg)
    results.write_cell_config(cfg.results_dir, chash, train_cfg)
    clash = sorted({k for k, h in done_hash.items() if h and h != chash})
    if clash:
        # Same (dataset, C, k, algorithm, seed) already recorded under DIFFERENT training settings
        # (e.g. another --cl_bias / --noise_eta / yaml in the same run_name): refuse instead of
        # silently skipping and mixing settings. Rows from before 2026-11-22 have no hash.
        raise RuntimeError(f'{len(clash)} cell(s) in {shard} were recorded with config_hash != {chash} '
                           f'(see {cfg.results_dir}/run_configs/); use a new --run_name. e.g. {clash[:3]}')

    print(f'GPU {cfg.gpu_id}/{cfg.num_gpus}  device={device}  dataset={cfg.dataset}  '
          f'epochs={cfg.epochs}  batch_size={cfg.batch_size}', flush=True)
    print(f'Algorithms: {my_algorithms}', flush=True)
    print(f'Resume: {len(done)} entries already in {shard}\n', flush=True)

    results.write_run_config(cfg.results_dir, {
        'run_name': cfg.run_name, 'algorithms': cfg.algorithms, 'dataset': cfg.dataset,
        'c_values': cfg.c_values, 'epochs': cfg.epochs, 'batch_size': cfg.batch_size,
        'seeds': cfg.seeds, 'gpu_id': cfg.gpu_id, 'num_gpus': cfg.num_gpus,
        'detail': cfg.detail, 'detail_log_every': cfg.detail_log_every,
        'tsne': cfg.tsne, 'tsne_every': cfg.tsne_every, 'tsne_max_points': cfg.tsne_max_points,
        'concentration': cfg.concentration, 'concentration_log_every': cfg.concentration_log_every,
        'knn_eval': cfg.knn_eval, 'knn_eval_k': cfg.knn_eval_k, 'knn_temperature': cfg.knn_temperature,
        'biasvariance': cfg.biasvariance, 'biasvariance_log_every': cfg.biasvariance_log_every,
        'biasvariance_m_resamples': cfg.biasvariance_m_resamples,
        'biasvariance_eval_size': cfg.biasvariance_eval_size,
        'only_q': cfg.only_q, 'only_bimodal': cfg.only_bimodal, 'batching': cfg.batching,
        'holdout_frac': cfg.holdout_frac, 'cl_bias': cfg.cl_bias, 'noise_eta': cfg.noise_eta,
        'switch_diag': cfg.switch_diag,
    })

    # Stashed in raw_cfg (like '_dataset_spec' below) rather than added to
    # every runner's uniform signature -- see src/pipeline/detail.py.
    cfg.raw['_detail'] = {
        'enabled': cfg.detail,
        'log_every': cfg.detail_log_every,
        'out_dir': os.path.join(cfg.results_dir, 'detail'),
        'tsne': {
            'enabled': cfg.tsne,
            'every': cfg.tsne_every,
            'max_points': cfg.tsne_max_points,
        },
        'concentration': {
            'enabled': cfg.concentration,
            'log_every': cfg.concentration_log_every,
        },
        'knn': {
            'enabled': cfg.knn_eval,
            'k': cfg.knn_eval_k,
            'temperature': cfg.knn_temperature,
        },
        'biasvariance': {
            'enabled': cfg.biasvariance,
            'log_every': cfg.biasvariance_log_every,
            'm_resamples': cfg.biasvariance_m_resamples,
            'eval_size': cfg.biasvariance_eval_size,
        },
        # With --seeds sweeping multiple seeds per cell, running the (fairly
        # expensive) diagnostic instrumentation for every seed would both
        # multiply the overhead and scatter per-seed curves into the same
        # output files (detail.cell_dir's paths aren't seed-scoped). Since
        # these diagnostics are for qualitative/exploratory inspection of one
        # representative training run -- not something meant to be averaged
        # across seeds the way final_accuracy is -- restrict them to a single
        # designated seed (the first one) regardless of how many seeds are
        # swept; final_accuracy is still recorded for every seed. See
        # detail._seed_matches.
        'diagnostics_seed': cfg.seeds[0],
    }

    # Fixed-class-count datasets (MNIST=10, CUB-200=200, ...) don't sweep C --
    # there's nothing to select a subset of, so c_values is forced to a
    # single-element list of the dataset's native class count.
    if spec is not None and spec.fixed_num_classes is not None:
        c_values = [spec.fixed_num_classes]
    else:
        c_values = [cfg.only_c] if cfg.only_c is not None else cfg.c_values

    for C in c_values:
        # Pre-ambiguous datasets (real candidate label sets from the source
        # paper) have no k to sweep -- k is meaningless when the candidate
        # set size is fixed by the real data, not chosen by us. Same for any
        # dataset whose generation isn't k-sized at all (sweeps_k=False, e.g.
        # cifar100-h's paper-faithful q-based generation). Both use a single
        # placeholder k=1 "cell" instead of the normal k-schedule.
        if cfg.only_bimodal is not None:
            k_values = [data.bimodal_sentinel_k(*cfg.only_bimodal)]
        elif cfg.only_q is not None:
            # --only_q replaces the whole k-schedule with one placeholder
            # cell. Real k is always a positive int in [1, C-1]; the negative
            # sentinel can never collide with a genuine k-based cell's dedup
            # key (see data.q_sentinel_k's docstring).
            k_values = [data.q_sentinel_k(cfg.only_q)]
        elif spec is not None and (spec.is_preambiguous or not spec.sweeps_k):
            k_values = cfg.only_k if cfg.only_k is not None else [1]
        else:
            k_values = cfg.only_k if cfg.only_k is not None else data.get_k_values(C)
        print(f'\n{"=" * 60}\ndataset={cfg.dataset}  C = {C}   '
              f'{"q = " + str(cfg.only_q) if cfg.only_q is not None else "k = " + str(k_values)}'
              f'\n{"=" * 60}', flush=True)

        # Seed is the OUTER loop, k the inner one (not the other way around):
        # this sweeps every k for the first seed to completion before
        # touching the second seed at all, so a full (all-k, all-algorithm)
        # single-seed pass is available for early/preliminary analysis
        # (merge + report can be run at any point mid-run -- only completed
        # cells show up) well before the remaining seeds finish accumulating
        # for the final mean/std. Resume/dedup is keyed on
        # (dataset, C, k, algorithm, seed) regardless of loop order, so this
        # doesn't change what eventually gets trained, only the order.
        for seed in cfg.seeds:
            for k in k_values:
                pending = [a for a in my_algorithms if (cfg.dataset, C, k, a, seed) not in done]
                if not pending:
                    print(f'\n  [skip] C={C} k={k} seed={seed}', flush=True)
                    continue

                print(f'\n--- C={C}  k={k}  seed={seed}  pending: {pending} ---', flush=True)
                _set_seed(seed)
                loaders, pl_ds, orig_targets = data.load_experiment_data(
                    C, k, cfg.data_dir, seed, cfg.log_dir, cfg.batch_size, dataset=cfg.dataset,
                    q=cfg.only_q, bimodal=cfg.only_bimodal, batching=cfg.batching,
                    holdout_frac=cfg.holdout_frac, cl_bias=cfg.cl_bias, noise_eta=cfg.noise_eta)

                # Stash the DatasetSpec (and, for --detail, the current k/seed)
                # in the raw config dict every runner already receives, so
                # create_model_for_spec / _IndexedDataset / detail.cell_dir
                # can pick the right backbone/transform/output path without
                # changing every runner's function signature.
                cfg.raw['_dataset_spec'] = spec
                # hptune (2026-11-12): eval_split: val -> every runner's loaders['test'] is the
                # --holdout_frac validation split (true labels), so the recorded final accuracy is
                # validation accuracy and the real test set is never looked at during tuning.
                if cfg.raw.get('eval_split', 'test') == 'val':
                    if 'holdout' not in loaders:
                        raise ValueError('eval_split: val needs --holdout_frac > 0')
                    from torch.utils.data import DataLoader, TensorDataset
                    _h = loaders['holdout']
                    loaders['test'] = DataLoader(TensorDataset(_h['x'], _h['y'].long()),
                                                 batch_size=cfg.batch_size, shuffle=False)
                # Under --only_q, `k` is q_sentinel_k's negative bookkeeping
                # value (real k is meaningless when candidate-set size is
                # drawn per-sample, not fixed) -- kept as-is for results.csv/
                # resume dedup above, which uses the loop's own `k` directly,
                # not this. But detail.cell_dir's directory naming and the
                # bias-variance diagnostic's fixed-size resampler (both keyed
                # off '_current_k') need a real positive int, so stash the
                # dataset's actual target MEAN candidate-set size instead --
                # exactly recoverable from q since generate_variable_pl_cl_datasets
                # includes each of the C-1 wrong classes independently w.p. q
                # (mean size = 1 + q*(C-1); round() only guards float noise).
                k_mean = k if cfg.only_q is None else round(1 + cfg.only_q * (C - 1))
                if cfg.only_bimodal is not None:
                    k_lo, k_hi, p_lo = cfg.only_bimodal
                    k_mean = round(p_lo * k_lo + (1 - p_lo) * k_hi)
                ctx = CellContext(results_dir=cfg.results_dir, dataset=cfg.dataset, C=C, k=k, k_mean=k_mean,
                                  seed=seed, dataset_spec=spec, switch_diag=cfg.switch_diag,
                                  detail=cfg.raw['_detail'], config_hash=chash)
                ctx.publish(cfg.raw)  # also mirrors the legacy raw_cfg['_current_k', ...] keys

                for alg in pending:
                    algo_spec = ALGORITHMS[alg]
                    hparams = {**ALGO_HPARAMS[alg], **(cfg.raw.get('hparams_override') or {})}
                    tag = f'GPU{cfg.gpu_id} {alg} C={C} k={k} seed={seed}'
                    print(f'\n  >> {alg}  C={C}  k={k}  seed={seed}', flush=True)

                    _set_seed(seed)
                    t0 = time.perf_counter()
                    acc = algo_spec.runner(loaders, pl_ds, orig_targets, C, hparams, cfg.raw,
                                            cfg.batch_size, cfg.epochs, device, tag, cfg.report_every)
                    elapsed = time.perf_counter() - t0

                    results.append_result(shard, cfg.dataset, C, k, alg, seed, acc, cfg.epochs, elapsed,
                                          config_hash=chash)
                    done.add((cfg.dataset, C, k, alg, seed))
                    print(f'  DONE  {alg}  seed={seed}  acc={acc:.2f}%  ({elapsed:.1f}s)', flush=True)

                del loaders, pl_ds, orig_targets
                gc.collect()
                torch.cuda.empty_cache()

                try:
                    results.merge_shards(cfg.results_dir)
                except Exception as e:
                    # merge_shards reads every sibling worker's shard file too --
                    # over NFS with several GPU processes writing concurrently, a
                    # transient torn read is possible. It must never take down
                    # this worker's training loop: the per-worker shard append
                    # above already safely recorded this cell's result, and the
                    # next successful merge (after the next cell) will catch up.
                    print(f'  [warn] merge_shards failed, continuing training '
                          f'(will retry after next cell): {e}', flush=True)

    print(f'\nGPU {cfg.gpu_id} finished. Results -> {cfg.results_dir}', flush=True)
