"""Random projection utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


ProjectionMethod = Literal["gaussian", "sparse", "achlioptas"]


@dataclass
class ProjectionConfig:
    enabled: bool
    method: ProjectionMethod
    dim: int
    chunk_size: int
    seed: int
    save_raw: bool = False


def project_vector(
    vec: torch.Tensor,
    cfg: ProjectionConfig,
) -> torch.Tensor:
    if not cfg.enabled:
        return vec
    gen = torch.Generator(device=vec.device)
    gen.manual_seed(cfg.seed)
    k = cfg.dim
    p = vec.numel()
    out = torch.zeros(k, device=vec.device, dtype=torch.float32)
    chunk = cfg.chunk_size
    for start in range(0, p, chunk):
        end = min(p, start + chunk)
        v_chunk = vec[start:end].to(torch.float32)
        r_chunk = _sample_projection(cfg.method, (k, end - start), gen, vec.device)
        out += r_chunk @ v_chunk
    scale = 1.0 / (cfg.dim ** 0.5)
    return out * scale


def _sample_projection(
    method: ProjectionMethod,
    shape: tuple[int, int],
    gen: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    if method == "gaussian":
        return torch.randn(shape, generator=gen, device=device)
    if method == "sparse":
        dense = torch.zeros(shape, device=device)
        mask = torch.rand(shape, generator=gen, device=device)
        dense[mask < 0.05] = 1.0
        dense[mask > 0.95] = -1.0
        return dense
    if method == "achlioptas":
        probs = torch.rand(shape, generator=gen, device=device)
        dense = torch.zeros(shape, device=device)
        dense[probs < 1.0 / 6.0] = 1.0
        dense[probs > 5.0 / 6.0] = -1.0
        return dense
    raise ValueError(f"Unknown projection method: {method}")
