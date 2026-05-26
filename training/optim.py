"""Optimizer helpers."""
from __future__ import annotations

from typing import Dict, Tuple

import torch


class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        after_scheduler: torch.optim.lr_scheduler._LRScheduler | None,
        last_epoch: int = -1,
    ) -> None:
        self.warmup_epochs = max(0, int(warmup_epochs))
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            scale = float(self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]
        if self.after_scheduler is None:
            return [base_lr for base_lr in self.base_lrs]
        if not self.finished:
            self.after_scheduler.base_lrs = self.base_lrs
            self.finished = True
        return self.after_scheduler.get_last_lr()

    def step(self, epoch: int | None = None) -> None:
        if self.last_epoch < self.warmup_epochs:
            super().step(epoch)
            return
        if self.after_scheduler is None:
            super().step(epoch)
            return
        if epoch is None:
            self.after_scheduler.step(None)
            self.last_epoch += 1
        else:
            self.after_scheduler.step(epoch - self.warmup_epochs)
            self.last_epoch = epoch
        self._last_lr = self.after_scheduler.get_last_lr()


def build_optimizer(
    params,
    lr: float,
    weight_decay: float,
    betas: Tuple[float, float],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, object],
    epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    name = str(cfg.get("name", "none")).lower()
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    if name in {"none", ""}:
        if warmup_epochs > 0:
            return WarmupScheduler(optimizer, warmup_epochs, None)
        return None
    if name == "cosine":
        t_max = int(cfg.get("t_max", epochs))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return WarmupScheduler(optimizer, warmup_epochs, scheduler) if warmup_epochs > 0 else scheduler
    if name == "step":
        step_size = int(cfg.get("step_size", 10))
        gamma = float(cfg.get("gamma", 0.1))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        return WarmupScheduler(optimizer, warmup_epochs, scheduler) if warmup_epochs > 0 else scheduler
    if name == "exponential":
        gamma = float(cfg.get("gamma", 0.95))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        return WarmupScheduler(optimizer, warmup_epochs, scheduler) if warmup_epochs > 0 else scheduler
    raise ValueError(f"Unknown scheduler: {name}")
