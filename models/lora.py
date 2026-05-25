"""LoRA utilities for linear layers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    enabled: bool
    r: int
    alpha: int
    dropout: float
    target: str


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float) -> None:
        super().__init__()
        self.base = base
        self.r = r
        self.alpha = alpha
        self.scale = alpha / max(1, r)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_b(self.lora_a(self.dropout(x))) * self.scale
        return base_out + lora_out


def _match_target(name: str, target: str) -> bool:
    if target == "all":
        return True
    if target == "attention":
        return "attn" in name or "attention" in name
    if target == "mlp":
        return "mlp" in name or "ffn" in name or "fc" in name
    return False


def apply_lora(model: nn.Module, cfg: LoRAConfig) -> nn.Module:
    if not cfg.enabled:
        return model
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _match_target(name, cfg.target):
            parent = _get_parent_module(model, name)
            if parent is None:
                continue
            child_name = name.split(".")[-1]
            setattr(parent, child_name, LoRALinear(module, cfg.r, cfg.alpha, cfg.dropout))
    return model


def _get_parent_module(model: nn.Module, name: str) -> nn.Module | None:
    parts = name.split(".")
    if len(parts) == 1:
        return None
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part, None)
        if parent is None:
            return None
    return parent


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield from module.lora_a.parameters()
            yield from module.lora_b.parameters()
