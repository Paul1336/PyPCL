"""Resolved configuration for a pipeline run: merges config.yaml with CLI args
into a single object passed around src/pipeline modules."""

import os
from dataclasses import dataclass, field

import yaml


@dataclass
class PipelineConfig:
    run_name: str
    algorithms: list
    dataset: str
    c_values: list
    epochs: int
    batch_size: int
    seed: int
    data_dir: str
    log_dir: str
    results_dir: str
    plots_dir: str
    gpu_id: int
    num_gpus: int
    algo_override: str
    report_every: int
    only_c: int
    only_k: int
    only_q: float
    raw: dict = field(default_factory=dict)   # full config.yaml (pico/comco/solar blocks)


def load_config(args) -> PipelineConfig:
    with open(args.config, encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    return PipelineConfig(
        run_name=args.run_name,
        algorithms=args.algorithms,
        dataset=getattr(args, 'dataset', 'cifar100-subset'),
        c_values=args.c_values,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        data_dir=args.data_dir,
        log_dir=args.log_dir,
        results_dir=os.path.join('results', args.run_name),
        plots_dir=os.path.join('plots', args.run_name),
        gpu_id=args.gpu_id,
        num_gpus=args.num_gpus,
        algo_override=args.algo,
        report_every=args.report_every,
        only_c=args.only_c,
        only_k=args.only_k,
        only_q=getattr(args, 'only_q', None),
        raw=raw,
    )
