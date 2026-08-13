"""Experiment loop: for each (C, k, algorithm) not already recorded, load
data, train, and append the result. Ties together gpu / data / algorithms /
results — this is the only module that orchestrates all of them."""

import gc
import time

import torch

from src.pipeline import data, results
from src.pipeline.algorithms import ALGORITHMS
from src.pipeline.algorithms.hparams import ALGO_HPARAMS
from src.pipeline.config import PipelineConfig
from src.pipeline.gpu import assign_algorithms, get_device


def run(cfg: PipelineConfig):
    device = get_device()
    my_algorithms = assign_algorithms(cfg.algorithms, cfg.gpu_id, cfg.num_gpus, cfg.algo_override)

    shard = results.shard_path(cfg.results_dir, cfg.gpu_id)
    done = results.load_done(shard)

    print(f'GPU {cfg.gpu_id}/{cfg.num_gpus}  device={device}  epochs={cfg.epochs}  '
          f'batch_size={cfg.batch_size}', flush=True)
    print(f'Algorithms: {my_algorithms}', flush=True)
    print(f'Resume: {len(done)} entries already in {shard}\n', flush=True)

    results.write_run_config(cfg.results_dir, {
        'run_name': cfg.run_name, 'algorithms': cfg.algorithms, 'c_values': cfg.c_values,
        'epochs': cfg.epochs, 'batch_size': cfg.batch_size, 'seed': cfg.seed,
        'gpu_id': cfg.gpu_id, 'num_gpus': cfg.num_gpus,
    })

    c_values = [cfg.only_c] if cfg.only_c is not None else cfg.c_values

    for C in c_values:
        k_values = [cfg.only_k] if cfg.only_k is not None else data.get_k_values(C)
        print(f'\n{"=" * 60}\nC = {C}   k = {k_values}\n{"=" * 60}', flush=True)

        for k in k_values:
            pending = [a for a in my_algorithms if (C, k, a) not in done]
            if not pending:
                print(f'  [skip] C={C} k={k}', flush=True)
                continue

            print(f'\n--- C={C}  k={k}  pending: {pending} ---', flush=True)
            loaders, pl_ds, orig_targets = data.load_experiment_data(
                C, k, cfg.data_dir, cfg.seed, cfg.log_dir, cfg.batch_size)

            for alg in pending:
                spec = ALGORITHMS[alg]
                hparams = ALGO_HPARAMS[alg]
                tag = f'GPU{cfg.gpu_id} {alg} C={C} k={k}'
                print(f'\n  >> {alg}  C={C}  k={k}', flush=True)

                t0 = time.perf_counter()
                acc = spec.runner(loaders, pl_ds, orig_targets, C, hparams, cfg.raw,
                                   cfg.batch_size, cfg.epochs, device, tag, cfg.report_every)
                elapsed = time.perf_counter() - t0

                results.append_result(shard, C, k, alg, cfg.seed, acc, cfg.epochs, elapsed)
                done.add((C, k, alg))
                print(f'  DONE  {alg}  acc={acc:.2f}%  ({elapsed:.1f}s)', flush=True)

            del loaders, pl_ds, orig_targets
            gc.collect()
            torch.cuda.empty_cache()

            results.merge_shards(cfg.results_dir)

    print(f'\nGPU {cfg.gpu_id} finished. Results -> {cfg.results_dir}', flush=True)
