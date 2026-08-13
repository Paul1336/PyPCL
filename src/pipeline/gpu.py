"""GPU/device selection and multi-GPU algorithm sharding.

Extracted from scripts/run_adam_comparison.py's gpu_id/num_gpus round-robin
logic so every entry point shares one implementation.
"""

import torch


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def assign_algorithms(all_algorithms: list, gpu_id: int, num_gpus: int, override: str = None) -> list:
    """Round-robin split of `all_algorithms` across `num_gpus` worker processes.

    `override` pins this worker to a single algorithm regardless of
    gpu_id/num_gpus (equivalent to the old scripts' --algo flag).
    """
    if override is not None:
        return [override]
    return [alg for i, alg in enumerate(all_algorithms) if i % num_gpus == gpu_id]
