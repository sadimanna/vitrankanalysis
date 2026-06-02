"""Gradient collection utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from utils.projection import ProjectionConfig, project_vector


@dataclass
class GradientCollectConfig:
    enabled: bool
    interval: int
    param_filter: str
    per_layer: bool
    attention_only: bool
    mlp_only: bool
    subsample: int
    normalization: str
    dtype: str
    cpu_offload: bool


class GradientCollector:
    def __init__(
        self,
        cfg: GradientCollectConfig,
        proj_cfg: ProjectionConfig,
    ) -> None:
        self.cfg = cfg
        self.proj_cfg = proj_cfg
        self.sample_idx: torch.Tensor | None = None

    def should_collect(self, step: int) -> bool:
        return self.cfg.enabled and (step % self.cfg.interval == 0)

    def collect(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        params = list(self._select_params(model))
        if self.cfg.per_layer:
            layer_map = self._group_by_layer(params)
            out: Dict[str, torch.Tensor] = {}
            for layer, pairs in layer_map.items():
                vec = self._flatten([p for _, p in pairs])
                raw_bytes = int(vec.numel() * vec.element_size())
                # print(
                #     f"gradient[{layer}] raw_len={vec.numel()} raw_mem={raw_bytes} bytes ({raw_bytes/1024**2:.3f} MB)"
                # )
                if self.proj_cfg.enabled and getattr(self.proj_cfg, "save_raw", False):
                    out[f"{layer}_raw"] = self._maybe_cast(vec)
                vec = self._apply_normalization(vec, mode="per_layer")
                vec = self._maybe_project(vec)
                proj_bytes = int(vec.numel() * vec.element_size())
                # print(
                #     f"gradient[{layer}] projected_len={vec.numel()} proj_mem={proj_bytes} bytes ({proj_bytes/1024**2:.3f} MB)"
                # )
                out[layer] = self._maybe_cast(vec)
            return out
        vec = self._flatten([p for _, p in params])
        raw_bytes = int(vec.numel() * vec.element_size())
        # print(
        #     f"gradient[all] raw_len={vec.numel()} raw_mem={raw_bytes} bytes ({raw_bytes/1024**2:.3f} MB)"
        # )
        out: Dict[str, torch.Tensor] = {}
        if self.proj_cfg.enabled and getattr(self.proj_cfg, "save_raw", False):
            out["all_raw"] = self._maybe_cast(vec)
        vec = self._apply_normalization(vec, mode=self.cfg.normalization)
        vec = self._maybe_project(vec)
        proj_bytes = int(vec.numel() * vec.element_size())
        # print(
        #     f"gradient[all] projected_len={vec.numel()} proj_mem={proj_bytes} bytes ({proj_bytes/1024**2:.3f} MB)"
        # )
        out["all"] = self._maybe_cast(vec)
        return out

    def _select_params(self, model: nn.Module) -> Iterable[Tuple[str, nn.Parameter]]:
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if self.cfg.attention_only and not self._is_attention(name):
                continue
            if self.cfg.mlp_only and not self._is_mlp(name):
                continue
            if self.cfg.param_filter == "layerwise" and not self._is_block_param(name):
                continue
            yield name, param

    def _group_by_layer(
        self, pairs: List[Tuple[str, nn.Parameter]]
    ) -> Dict[str, List[Tuple[str, nn.Parameter]]]:
        layer_map: Dict[str, List[Tuple[str, nn.Parameter]]] = {}
        for name, param in pairs:
            key = self._layer_key(name)
            layer_map.setdefault(key, []).append((name, param))
        return layer_map

    def _layer_key(self, name: str) -> str:
        if "blocks" in name:
            parts = name.split("blocks")
            suffix = parts[1].split(".")[1]
            return f"blocks.{suffix}"
        if "encoder.layers" in name:
            parts = name.split("encoder.layers")
            suffix = parts[1].split(".")[1]
            return f"encoder.layers.{suffix}"
        return "other"

    def _is_block_param(self, name: str) -> bool:
        return "blocks" in name or "encoder.layers" in name

    def _is_attention(self, name: str) -> bool:
        return "attn" in name or "attention" in name

    def _is_mlp(self, name: str) -> bool:
        return "mlp" in name or "ffn" in name or "fc" in name

    def _flatten(self, params: Iterable[nn.Parameter]) -> torch.Tensor:
        grads = [p.grad.detach().reshape(-1) for p in params]
        vec = torch.cat(grads, dim=0)
        if self.cfg.subsample > 0:
            if self.sample_idx is None:
                idx = torch.randperm(vec.numel(), device=vec.device)[: self.cfg.subsample]
                self.sample_idx = idx
            vec = vec[self.sample_idx]
        return vec

    def _apply_normalization(self, vec: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "none":
            return vec
        if mode == "l2":
            denom = torch.norm(vec) + 1e-12
            return vec / denom
        if mode == "unit_fro":
            denom = torch.norm(vec) + 1e-12
            return vec / denom
        if mode == "per_layer":
            denom = torch.norm(vec) + 1e-12
            return vec / denom
        return vec

    def _maybe_project(self, vec: torch.Tensor) -> torch.Tensor:
        return project_vector(vec, self.proj_cfg) if self.proj_cfg.enabled else vec

    def _maybe_cast(self, vec: torch.Tensor) -> torch.Tensor:
        dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
        target_dtype = dtype_map.get(self.cfg.dtype, torch.float32)
        if self.cfg.cpu_offload:
            return vec.to(target_dtype).cpu()
        return vec.to(target_dtype)
