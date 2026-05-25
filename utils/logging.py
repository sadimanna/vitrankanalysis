"""Logging utilities."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict


def setup_logger(name: str, output_dir: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(Path(output_dir) / "train.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console)
    return logger


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
