"""Experiment loop: for each (C, k, algorithm) not already recorded, load
data, train, and append the result. Ties together gpu / data / algorithms /
results — this is the only module that orchestrates all of them."""

import gc
import os
import time

import torch

from src.pipeline import data, results
from src.pipeline.algorithms import ALGORITHMS
from src.pipeline.algorithms.hparams import ALGO_HPARAMS
from src.pipeline.config import PipelineConfig
from src.pipeline.gpu import assign_algorithms, get_device

# Algorithms that need real RGB image weak/strong augmentation pairs
# (loaders['pico'] / loaders['comco']) -- unsupported on any DatasetSpec with
# supports_pico_family=False (grayscale, tabular, preambiguous-feature, or
# lazy-path datasets). See docs/00_paper_alignment_guide.md.
IMAGE_ONLY_ALGORITHMS = {'PiCO', 'PiCO-Oracle', 'PiCO-MOCO', 'PiCO-Fixed', 'PiCO-MCL', 'PiCO-SC', 'ComCo', 'ComCo-Fixed', 'SoLar'}


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
    done = results.load_done(shard)

    print(f'GPU {cfg.gpu_id}/{cfg.num_gpus}  device={device}  dataset={cfg.dataset}  '
          f'epochs={cfg.epochs}  batch_size={cfg.batch_size}', flush=True)
    print(f'Algorithms: {my_algorithms}', flush=True)
    print(f'Resume: {len(done)} entries already in {shard}\n', flush=True)

    results.write_run_config(cfg.results_dir, {
        'run_name': cfg.run_name, 'algorithms': cfg.algorithms, 'dataset': cfg.dataset,
        'c_values': cfg.c_values, 'epochs': cfg.epochs, 'batch_size': cfg.batch_size,
        'seed': cfg.seed, 'gpu_id': cfg.gpu_id, 'num_gpus': cfg.num_gpus,
        'detail': cfg.detail, 'detail_log_every': cfg.detail_log_every,
        'tsne': cfg.tsne, 'tsne_every': cfg.tsne_every, 'tsne_max_points': cfg.tsne_max_points,
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
        if cfg.only_q is not None:
            # --only_q replaces the whole k-schedule with one placeholder
            # cell. Real k is always a positive int in [1, C-1]; the negative
            # sentinel can never collide with a genuine k-based cell's dedup
            # key (see data.q_sentinel_k's docstring).
            k_values = [data.q_sentinel_k(cfg.only_q)]
        elif spec is not None and (spec.is_preambiguous or not spec.sweeps_k):
            k_values = [cfg.only_k] if cfg.only_k is not None else [1]
        else:
            k_values = [cfg.only_k] if cfg.only_k is not None else data.get_k_values(C)
        print(f'\n{"=" * 60}\ndataset={cfg.dataset}  C = {C}   '
              f'{"q = " + str(cfg.only_q) if cfg.only_q is not None else "k = " + str(k_values)}'
              f'\n{"=" * 60}', flush=True)

        for k in k_values:
            pending = [a for a in my_algorithms if (cfg.dataset, C, k, a) not in done]
            if not pending:
                print(f'  [skip] C={C} k={k}', flush=True)
                continue

            print(f'\n--- C={C}  k={k}  pending: {pending} ---', flush=True)
            loaders, pl_ds, orig_targets = data.load_experiment_data(
                C, k, cfg.data_dir, cfg.seed, cfg.log_dir, cfg.batch_size, dataset=cfg.dataset,
                q=cfg.only_q)

            # Stash the DatasetSpec (and, for --detail, the current k) in the
            # raw config dict every runner already receives, so
            # create_model_for_spec / _IndexedDataset / detail.cell_dir can
            # pick the right backbone/transform/output path without changing
            # every runner's function signature.
            cfg.raw['_dataset_spec'] = spec
            cfg.raw['_current_k'] = k

            for alg in pending:
                algo_spec = ALGORITHMS[alg]
                hparams = ALGO_HPARAMS[alg]
                tag = f'GPU{cfg.gpu_id} {alg} C={C} k={k}'
                print(f'\n  >> {alg}  C={C}  k={k}', flush=True)

                t0 = time.perf_counter()
                acc = algo_spec.runner(loaders, pl_ds, orig_targets, C, hparams, cfg.raw,
                                        cfg.batch_size, cfg.epochs, device, tag, cfg.report_every)
                elapsed = time.perf_counter() - t0

                results.append_result(shard, cfg.dataset, C, k, alg, cfg.seed, acc, cfg.epochs, elapsed)
                done.add((cfg.dataset, C, k, alg))
                print(f'  DONE  {alg}  acc={acc:.2f}%  ({elapsed:.1f}s)', flush=True)

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
