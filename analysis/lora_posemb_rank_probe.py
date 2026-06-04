"""Probe positional-embedding rank through LoRA adapters."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.lora import LoRAConfig, LoRALinear
from models.vit_factory import create_vit
from utils.config import load_config
from utils.projection import ProjectionConfig, project_vector
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set zero image inputs, optionally zero LoRA base layers, initialize LoRA "
            "A/B uniformly, and measure ranks of positional-embedding gradients."
        )
    )
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--source", default="timm")
    parser.add_argument("--model", default="vit_small_patch16_224")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--probes", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=128)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lora-target", default=None, choices=["all", "attention", "mlp"])
    parser.add_argument("--uniform-low", type=float, default=-0.02)
    parser.add_argument("--uniform-high", type=float, default=0.02)
    parser.add_argument("--projection-dim", type=int, default=8192)
    parser.add_argument(
        "--projection-method",
        default="gaussian",
        choices=["gaussian", "sparse", "achlioptas"],
    )
    parser.add_argument("--projection-seed", type=int, default=123)
    parser.add_argument("--projection-chunk-size", type=int, default=16384)
    parser.add_argument("--rank-tol", type=float, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--keep-lora-base", action="store_true")
    parser.add_argument("--keep-patch-embed", action="store_true")
    parser.add_argument("--keep-class-token", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def cfg_get(cfg: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    value: Any = cfg
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def find_positional_embedding_params(
    model: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    candidates = []
    tokens = ("pos_embed", "pos_embedding", "position_embedding", "positional_embedding")
    for name, param in model.named_parameters():
        lower = name.lower()
        if any(token in lower for token in tokens):
            candidates.append((name, param))
    return candidates


def find_token_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    candidates = []
    tokens = ("cls_token", "class_token", "dist_token")
    for name, param in model.named_parameters():
        lower = name.lower()
        if any(token in lower for token in tokens):
            candidates.append((name, param))
    return candidates


def infer_input_channels(model: nn.Module) -> int:
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and (
            "patch_embed" in name or "conv_proj" in name
        ):
            return int(module.in_channels)
    return 3


def init_lora_uniform(model: nn.Module, low: float, high: float) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            nn.init.uniform_(module.lora_a.weight, low, high)
            nn.init.uniform_(module.lora_b.weight, low, high)
            count += 1
    return count


def zero_lora_base_layers(model: nn.Module) -> int:
    count = 0
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.base.weight.zero_()
                if module.base.bias is not None:
                    module.base.bias.zero_()
                count += 1
    return count


def zero_patch_embedding(model: nn.Module) -> list[str]:
    zeroed = []
    with torch.no_grad():
        for name, module in model.named_modules():
            is_patch_module = "patch_embed" in name or "conv_proj" in name
            if is_patch_module and isinstance(module, (nn.Conv2d, nn.Linear)):
                module.weight.zero_()
                if module.bias is not None:
                    module.bias.zero_()
                zeroed.append(name)
    return zeroed


def zero_named_params(params: list[tuple[str, nn.Parameter]]) -> list[str]:
    zeroed = []
    with torch.no_grad():
        for name, param in params:
            param.zero_()
            zeroed.append(name)
    return zeroed


def flatten_positional_grads(
    pos_params: list[tuple[str, nn.Parameter]],
) -> torch.Tensor:
    grads = []
    for name, param in pos_params:
        if param.grad is None:
            raise RuntimeError(f"No gradient found for positional embedding: {name}")
        grads.append(param.grad.detach().reshape(-1).to(torch.float32).cpu())
    return torch.cat(grads, dim=0)


def rank_metrics(samples: torch.Tensor, tol: float | None = None) -> dict[str, Any]:
    if samples.ndim != 2:
        raise ValueError("rank_metrics expects a 2D matrix")
    matrix = samples.to(torch.float64)
    gram = matrix @ matrix.T
    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0.0)
    singular_values = torch.sqrt(eigvals).flip(0)
    if singular_values.numel() == 0:
        return {
            "shape": list(samples.shape),
            "rank": 0,
            "tol": 0.0,
            "stable_rank": 0.0,
            "effective_rank": 0.0,
            "singular_values": [],
        }
    max_sv = float(singular_values.max().item())
    if tol is None:
        eps_dtype = samples.dtype if samples.is_floating_point() else torch.float32
        eps = torch.finfo(eps_dtype).eps
        tol = max(samples.shape) * eps * max_sv
    nonzero = singular_values[singular_values > tol]
    squared = singular_values.square()
    stable_rank = 0.0
    if max_sv > 0:
        stable_rank = float(squared.sum().item() / (max_sv * max_sv))
    effective_rank = 0.0
    total = float(singular_values.sum().item())
    if total > 0:
        probs = singular_values / total
        entropy = -(probs[probs > 0] * torch.log(probs[probs > 0])).sum()
        effective_rank = float(torch.exp(entropy).item())
    return {
        "shape": list(samples.shape),
        "rank": int(nonzero.numel()),
        "tol": float(tol),
        "stable_rank": stable_rank,
        "effective_rank": effective_rank,
        "singular_values": [float(v) for v in singular_values[:20].tolist()],
    }


def tensor_token_matrix(param: torch.Tensor) -> torch.Tensor:
    tensor = param.detach().to(torch.float32).cpu()
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(-1, tensor.shape[-1])


def direct_lora_posemb_ranks(
    model: nn.Module,
    pos_params: list[tuple[str, nn.Parameter]],
    tol: float | None,
) -> list[dict[str, Any]]:
    if not pos_params:
        return []
    pos_matrix = tensor_token_matrix(pos_params[0][1])
    ranks = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            if module.base.in_features != pos_matrix.shape[-1]:
                continue
            device = module.lora_a.weight.device
            x = pos_matrix.to(device)
            y = module.lora_b(module.lora_a(x)) * module.scale
            ranks.append(
                {
                    "module": name,
                    "input_features": int(module.base.in_features),
                    "output_features": int(module.base.out_features),
                    "rank": rank_metrics(y.cpu(), tol)["rank"],
                    "shape": list(y.shape),
                }
            )
    return ranks


def summarize_module_ranks(ranks: list[dict[str, Any]]) -> dict[str, Any]:
    if not ranks:
        return {"count": 0}
    values = torch.tensor([item["rank"] for item in ranks], dtype=torch.float32)
    return {
        "count": len(ranks),
        "min": int(values.min().item()),
        "max": int(values.max().item()),
        "mean": float(values.mean().item()),
        "first_modules": ranks[:10],
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg_get(cfg, ("seed",), 42))
    set_seed(seed, deterministic=True)

    data_cfg = cfg_get(cfg, ("training", "dataset"), {})
    model_cfg = cfg_get(cfg, ("model",), {})
    config_lora = dict(cfg_get(cfg, ("model", "lora"), {}))

    source = args.source or model_cfg.get("source", "timm")
    model_name = args.model or model_cfg.get("name", "vit_base_patch16_224")
    image_size = args.image_size or data_cfg.get("image_size", 224)
    patch_size = args.patch_size or data_cfg.get("patch_size")
    num_classes = args.num_classes or data_cfg.get("num_classes", 10)

    config_lora["enabled"] = True
    if args.lora_r is not None:
        config_lora["r"] = args.lora_r
    if args.lora_alpha is not None:
        config_lora["alpha"] = args.lora_alpha
    config_lora["dropout"] = args.lora_dropout
    if args.lora_target is not None:
        config_lora["target"] = args.lora_target
    lora_cfg = LoRAConfig(**config_lora)

    device = resolve_device(args.device)

    model = create_vit(
        source=source,
        name=model_name,
        pretrained=args.pretrained,
        num_classes=int(num_classes),
        lora_cfg=lora_cfg,
        image_size=int(image_size) if image_size is not None else None,
        patch_size=int(patch_size) if patch_size is not None else None,
    ).to(device)
    model.eval()

    lora_count = init_lora_uniform(model, args.uniform_low, args.uniform_high)
    zeroed_base_count = 0 if args.keep_lora_base else zero_lora_base_layers(model)
    zeroed_patch_modules = [] if args.keep_patch_embed else zero_patch_embedding(model)

    pos_params = find_positional_embedding_params(model)
    if not pos_params:
        raise RuntimeError("No positional embedding parameter found in the model")
    for _, param in pos_params:
        param.requires_grad_(True)

    token_params = find_token_params(model)
    zeroed_token_params = [] if args.keep_class_token else zero_named_params(token_params)

    proj_cfg = ProjectionConfig(
        enabled=args.projection_dim > 0,
        method=args.projection_method,
        dim=args.projection_dim,
        chunk_size=args.projection_chunk_size,
        seed=args.projection_seed,
    )

    in_channels = infer_input_channels(model)
    raw_grads = []
    projected_grads = []
    label_gen = torch.Generator(device="cpu")
    label_gen.manual_seed(seed + 1)

    grad_ranks = []
    grad_effective_ranks = []

    raw_grads = []
    projected_grads = []

    for _ in range(args.probes):
        model.zero_grad(set_to_none=True)

        images = torch.zeros(
            args.batch_size,
            in_channels,
            int(image_size),
            int(image_size),
            device=device,
        )

        logits = model(images)

        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        classes = int(logits.shape[-1])

        labels = torch.randint(
            low=0,
            high=classes,
            size=(args.batch_size,),
            generator=label_gen,
            device="cpu",
        ).to(device)

        loss = F.cross_entropy(logits, labels)
        loss.backward()

        # --------------------------------------------------
        # positional gradient matrix G_P
        # --------------------------------------------------

        name, pos_param = pos_params[0]

        if pos_param.grad is None:
            raise RuntimeError(
                f"No gradient found for positional embedding: {name}"
            )

        pos_grad = tensor_token_matrix(pos_param.grad)

        # matrix rank of T x d gradient
        rank = int(torch.linalg.matrix_rank(pos_grad).item())

        s = torch.linalg.svdvals(pos_grad)
        # print(s[:20].cpu().numpy())
        # print((s**2 / (s**2).sum()).cpu().numpy()[:20])

        if torch.sum(s**2) > 0:
            energy = torch.cumsum(s**2, dim=0) / torch.sum(s**2)
            effective_rank = (
                torch.searchsorted(
                    energy,
                    torch.tensor(0.99, device=energy.device),
                ).item()
                + 1
            )
        else:
            effective_rank = 0

        grad_ranks.append(rank)
        grad_effective_ranks.append(int(effective_rank))

        # --------------------------------------------------
        # flattened gradient for probe-space analysis
        # --------------------------------------------------

        flat_grad = pos_grad.reshape(-1).cpu()

        raw_grads.append(flat_grad)

        projected_grads.append(
            project_vector(flat_grad, proj_cfg).detach().cpu()
        )

    raw_grad_matrix = torch.stack(raw_grads, dim=0)
    projected_grad_matrix = torch.stack(projected_grads, dim=0)

    pos_matrix = tensor_token_matrix(pos_params[0][1])

    direct_ranks = direct_lora_posemb_ranks(
        model,
        pos_params,
        args.rank_tol,
    )

    result = {
        "positional_embeddings": {
        "params": [
            {
                "name": name,
                "shape": list(param.shape),
            }
            for name, param in pos_params
        ],
        "token_matrix_rank": rank_metrics(
            pos_matrix,
            args.rank_tol,
        ),
    },

    "direct_lora_posemb_projection": summarize_module_ranks(
        direct_ranks
    ),

    "gradient_rank": {
        "per_probe_matrix_rank": {
            "mean": float(sum(grad_ranks) / len(grad_ranks)),
            "min": int(min(grad_ranks)),
            "max": int(max(grad_ranks)),
            "values": grad_ranks,
        },
        "per_probe_effective_rank": {
            "mean": float(
                sum(grad_effective_ranks)
                / len(grad_effective_ranks)
            ),
            "min": int(min(grad_effective_ranks)),
            "max": int(max(grad_effective_ranks)),
            "values": grad_effective_ranks,
        },
        "probe_space_rank": rank_metrics(
            raw_grad_matrix,
            args.rank_tol,
        ),
        "projected_probe_space_rank": rank_metrics(
            projected_grad_matrix,
            args.rank_tol,
        ),
    },
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
