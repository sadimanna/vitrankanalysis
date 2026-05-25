"""ViT factory utilities."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

try:
    import timm
except Exception:  # pragma: no cover - optional dependency
    timm = None

try:
    import torchvision
    from torchvision.models.vision_transformer import VisionTransformer
except Exception:  # pragma: no cover - optional dependency
    torchvision = None
    VisionTransformer = None

from models.lora import LoRAConfig, apply_lora


def create_vit(
    source: str,
    name: str,
    pretrained: bool,
    num_classes: int,
    lora_cfg: LoRAConfig,
) -> nn.Module:
    if source == "timm":
        if timm is None:
            raise RuntimeError("timm is not available")
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    elif source == "torchvision":
        if torchvision is None:
            raise RuntimeError("torchvision is not available")
        model = _create_torchvision_vit(name, pretrained, num_classes)
    else:
        raise ValueError(f"Unknown model source: {source}")
    model = apply_lora(model, lora_cfg)
    return model


def _create_torchvision_vit(name: str, pretrained: bool, num_classes: int) -> nn.Module:
    if name == "vit_b_16":
        model = torchvision.models.vit_b_16(pretrained=pretrained)
    elif name == "vit_l_16":
        model = torchvision.models.vit_l_16(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported torchvision ViT: {name}")
    if hasattr(model, "heads"):
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)
    return model


def get_vit_block_names(model: nn.Module) -> Tuple[str, ...]:
    names = []
    for name, module in model.named_modules():
        if "blocks" in name or "encoder" in name:
            if name.endswith("."):
                names.append(name[:-1])
    if not names:
        for name, module in model.named_modules():
            if "block" in name:
                names.append(name)
    return tuple(sorted(set(names)))
