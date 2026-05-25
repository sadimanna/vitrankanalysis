"""Optimizer helpers."""
from __future__ import annotations

from typing import Tuple

import torch


def build_optimizer(
    params,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
