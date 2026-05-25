"""Distributed helpers."""
from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def init_distributed() -> Tuple[bool, int]:
    if "RANK" not in os.environ:
        return False, 0
    if dist.is_initialized():
        return True, dist.get_rank()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    return True, rank


def is_rank0(rank: int) -> bool:
    return rank == 0
