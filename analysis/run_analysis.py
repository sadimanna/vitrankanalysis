"""Offline analysis from saved Gram matrices."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis.spectral import compute_summary, eigenvalues_from_gram
from utils.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gram", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.9, 0.95, 0.99])
    parser.add_argument("--powerlaw", action="store_true")
    args = parser.parse_args()

    K = np.load(args.gram)
    eigenvalues = eigenvalues_from_gram(K)
    summary = compute_summary(eigenvalues, args.thresholds, args.powerlaw)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "eigenvalues.npy", summary.eigenvalues)
    write_json(out_dir / "spectral_summary.json", summary.__dict__)


if __name__ == "__main__":
    main()
