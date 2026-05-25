"""Spectral analysis utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class SpectralSummary:
    eigenvalues: np.ndarray
    effective_rank: float
    participation_ratio: float
    stable_rank: float
    spectral_entropy: float
    energy_thresholds: Dict[str, int]
    powerlaw_slope: float | None


def eigenvalues_from_gram(K: np.ndarray) -> np.ndarray:
    if K.size == 0:
        return np.array([])
    vals = np.linalg.eigvalsh(K)
    vals = np.maximum(vals, 0.0)
    return vals[::-1]


def compute_summary(
    eigenvalues: np.ndarray,
    energy_thresholds: List[float],
    fit_powerlaw: bool,
) -> SpectralSummary:
    if eigenvalues.size == 0:
        return SpectralSummary(
            eigenvalues=eigenvalues,
            effective_rank=0.0,
            participation_ratio=0.0,
            stable_rank=0.0,
            spectral_entropy=0.0,
            energy_thresholds={str(t): 0 for t in energy_thresholds},
            powerlaw_slope=None,
        )
    lam = eigenvalues
    total = lam.sum()
    p = lam / max(total, 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    effective_rank = float(np.exp(entropy))
    participation_ratio = float((lam.sum() ** 2) / (np.sum(lam ** 2) + 1e-12))
    stable_rank = float(lam.sum() / (lam.max() + 1e-12))
    cumulative = np.cumsum(lam) / max(total, 1e-12)
    thresholds = {str(t): int(np.searchsorted(cumulative, t) + 1) for t in energy_thresholds}
    slope = _fit_powerlaw(lam) if fit_powerlaw else None
    return SpectralSummary(
        eigenvalues=lam,
        effective_rank=effective_rank,
        participation_ratio=participation_ratio,
        stable_rank=stable_rank,
        spectral_entropy=float(entropy),
        energy_thresholds=thresholds,
        powerlaw_slope=slope,
    )


def _fit_powerlaw(lam: np.ndarray) -> float | None:
    if lam.size < 5:
        return None
    ranks = np.arange(1, lam.size + 1)
    tail = lam[lam > 0]
    if tail.size < 5:
        return None
    x = np.log(ranks[: tail.size])
    y = np.log(tail)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)
