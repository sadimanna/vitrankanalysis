"""Subspace analysis utilities."""
from __future__ import annotations

from typing import Tuple

import numpy as np


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    q_u, _ = np.linalg.qr(U)
    q_v, _ = np.linalg.qr(V)
    sigma = np.linalg.svd(q_u.T @ q_v, compute_uv=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    return np.arccos(sigma)


def subspace_overlap(U: np.ndarray, V: np.ndarray) -> float:
    angles = principal_angles(U, V)
    return float(np.mean(np.cos(angles) ** 2))
