"""Temporal Gram matrix utilities."""
from __future__ import annotations

from typing import List

import torch


class GramAccumulator:
    def __init__(self) -> None:
        self.vectors: List[torch.Tensor] = []
        self.K: torch.Tensor | None = None

    def add(self, vec: torch.Tensor) -> None:
        vec = vec.to(torch.float32).flatten()
        if self.K is None:
            self.K = torch.zeros((1, 1), dtype=torch.float32)
            self.K[0, 0] = torch.dot(vec, vec)
        else:
            prev = torch.stack(self.vectors, dim=0)
            dots = prev @ vec
            new_row = torch.cat([dots, vec.new_tensor([torch.dot(vec, vec)])])
            K_new = torch.zeros(
                (self.K.shape[0] + 1, self.K.shape[1] + 1), dtype=torch.float32
            )
            K_new[:-1, :-1] = self.K
            K_new[-1, :-1] = new_row[:-1]
            K_new[:-1, -1] = new_row[:-1]
            K_new[-1, -1] = new_row[-1]
            self.K = K_new
        self.vectors.append(vec)

    def add_batch(self, batch: torch.Tensor) -> None:
        for vec in batch:
            self.add(vec)

    def get_matrix(self) -> torch.Tensor:
        if self.K is None:
            return torch.empty((0, 0), dtype=torch.float32)
        return self.K

    def get_vectors(self) -> torch.Tensor:
        if not self.vectors:
            return torch.empty((0, 0), dtype=torch.float32)
        return torch.stack(self.vectors, dim=0)
