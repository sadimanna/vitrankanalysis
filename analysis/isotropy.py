"""Isotropy analysis utilities."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch


def isotropy_metrics(weight: torch.Tensor) -> Dict[str, float]:
    w = weight.detach().to(torch.float32)
    if w.ndim == 1:
        w = w.unsqueeze(0)
    u, s, v = torch.linalg.svd(w, full_matrices=False)
    s_np = s.cpu().numpy()
    cond = float(s_np.max() / max(s_np.min(), 1e-12))
    spread = float(s_np.max() - s_np.min())
    effective_dim = float((s_np.sum() ** 2) / (np.sum(s_np ** 2) + 1e-12))
    return {
        "condition_number": cond,
        "singular_spread": spread,
        "effective_dim": effective_dim,
    }
