"""Streaming PCA utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class OjaConfig:
    components: int
    lr: float


class OjaTracker:
    def __init__(self, cfg: OjaConfig, dim: int) -> None:
        self.cfg = cfg
        self.components = torch.randn(dim, cfg.components) / (dim ** 0.5)

    def update(self, vec: torch.Tensor) -> None:
        x = vec.to(torch.float32)
        y = x @ self.components
        update = torch.ger(x, y) - self.components @ (y.unsqueeze(0) * y.unsqueeze(1))
        self.components += self.cfg.lr * update
        self.components = torch.linalg.qr(self.components, mode="reduced").Q


@dataclass
class IncrementalPCAConfig:
    components: int
    batch_size: int


class IncrementalPCA:
    def __init__(self, cfg: IncrementalPCAConfig, dim: int) -> None:
        self.cfg = cfg
        self.dim = dim
        self.buffer = []
        self.components: Optional[torch.Tensor] = None
        self.mean = torch.zeros(dim)
        self.count = 0

    def partial_fit(self, vec: torch.Tensor) -> None:
        self.buffer.append(vec.to(torch.float32))
        if len(self.buffer) >= self.cfg.batch_size:
            self._update()

    def _update(self) -> None:
        batch = torch.stack(self.buffer, dim=0)
        self.count += batch.shape[0]
        batch_mean = batch.mean(dim=0)
        self.mean = self.mean + (batch_mean - self.mean) * (batch.shape[0] / max(self.count, 1))
        centered = batch - self.mean
        _, _, v = torch.linalg.svd(centered, full_matrices=False)
        self.components = v[: self.cfg.components].T
        self.buffer = []


class FrequentDirections:
    def __init__(self, sketch_dim: int, dim: int) -> None:
        self.sketch_dim = sketch_dim
        self.dim = dim
        self.B = torch.zeros(sketch_dim, dim)
        self.next_row = 0

    def update(self, vec: torch.Tensor) -> None:
        self.B[self.next_row] = vec.to(torch.float32)
        self.next_row += 1
        if self.next_row >= self.sketch_dim:
            self._compress()
            self.next_row = self.sketch_dim // 2

    def _compress(self) -> None:
        u, s, v = torch.linalg.svd(self.B, full_matrices=False)
        shrink = s[self.sketch_dim // 2] ** 2
        s_shrunk = torch.sqrt(torch.clamp(s ** 2 - shrink, min=0.0))
        self.B = torch.diag(s_shrunk) @ v

    def get_components(self, k: int) -> torch.Tensor:
        u, s, v = torch.linalg.svd(self.B, full_matrices=False)
        return v[:k].T
