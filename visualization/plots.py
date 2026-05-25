"""Plotting utilities for spectral analysis."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


def set_style(style: str) -> None:
    if style == "paper":
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update({
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        })


def plot_scree(eigenvalues: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(eigenvalues, marker="o", linewidth=1.5)
    plt.title("Scree Plot")
    plt.xlabel("Component")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_log_decay(eigenvalues: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(np.log(eigenvalues + 1e-12), linewidth=1.5)
    plt.title("Log Eigenvalue Decay")
    plt.xlabel("Component")
    plt.ylabel("Log Eigenvalue")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_cumulative_energy(eigenvalues: np.ndarray, out_path: Path) -> None:
    cumulative = np.cumsum(eigenvalues) / max(eigenvalues.sum(), 1e-12)
    plt.figure(figsize=(6, 4))
    plt.plot(cumulative, linewidth=1.5)
    plt.title("Cumulative Energy")
    plt.xlabel("Components")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_effective_rank_over_time(values: Iterable[float], out_path: Path) -> None:
    vals = list(values)
    plt.figure(figsize=(6, 4))
    plt.plot(vals, linewidth=1.5)
    plt.title("Effective Rank Over Time")
    plt.xlabel("Snapshot")
    plt.ylabel("Effective Rank")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_layer_heatmap(metric_map: Dict[str, List[float]], out_path: Path) -> None:
    layers = list(metric_map.keys())
    data = np.array([metric_map[layer] for layer in layers])
    plt.figure(figsize=(8, max(3, len(layers) * 0.3)))
    plt.imshow(data, aspect="auto", cmap="viridis")
    plt.colorbar(label="Metric")
    plt.yticks(range(len(layers)), layers)
    plt.xlabel("Snapshot")
    plt.title("Layerwise Metric Heatmap")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
