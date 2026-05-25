"""Training entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
import torch
from torch.cuda.amp import GradScaler, autocast

from analysis.gram import GramAccumulator
from analysis.gradient_collector import GradientCollectConfig, GradientCollector
from analysis.isotropy import isotropy_metrics
from analysis.spectral import compute_summary, eigenvalues_from_gram
from analysis.streaming import FrequentDirections, IncrementalPCA, IncrementalPCAConfig, OjaConfig, OjaTracker
from models.lora import LoRAConfig, lora_parameters
from models.vit_factory import create_vit
from training.data import get_dataloaders
from training.optim import build_optimizer
from utils.config import load_config
from utils.dist import init_distributed, is_rank0
from utils.logging import setup_logger, write_json
from utils.projection import ProjectionConfig
from utils.seed import set_seed
from visualization.plots import (
    plot_cumulative_energy,
    plot_effective_rank_over_time,
    plot_log_decay,
    plot_scree,
    plot_layer_heatmap,
    set_style,
)


def train_loop(cfg: Dict) -> None:
    distributed, rank = init_distributed()
    set_seed(cfg["seed"], deterministic=False)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics").mkdir(exist_ok=True)
    (output_dir / "spectra").mkdir(exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    (output_dir / "projections").mkdir(exist_ok=True)

    with open(output_dir / "config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    logger = setup_logger("train", str(output_dir))

    train_cfg = cfg["training"]
    data_cfg = train_cfg["dataset"]

    train_loader, _, num_classes = get_dataloaders(
        data_cfg["name"],
        data_cfg["data_dir"],
        data_cfg["image_size"],
        train_cfg["batch_size"],
        data_cfg["num_workers"],
    )

    lora_cfg = LoRAConfig(**cfg["model"]["lora"])
    model = create_vit(
        cfg["model"]["source"],
        cfg["model"]["name"],
        cfg["model"]["pretrained"],
        num_classes,
        lora_cfg,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if lora_cfg.enabled:
        optimizer_params = list(lora_parameters(model))
    else:
        optimizer_params = model.parameters()

    optimizer = build_optimizer(
        optimizer_params,
        train_cfg["lr"],
        train_cfg["weight_decay"],
        tuple(train_cfg["betas"]),
    )

    scaler = GradScaler(enabled=train_cfg["mixed_precision"])
    criterion = torch.nn.CrossEntropyLoss()

    grad_cfg = GradientCollectConfig(**train_cfg["grad_collect"])
    proj_cfg = ProjectionConfig(**train_cfg["projection"])
    collector = GradientCollector(grad_cfg, proj_cfg)

    gram_map: Dict[str, GramAccumulator] = {}
    eff_rank_series: List[float] = []
    layer_rank_series: Dict[str, List[float]] = {}

    oja = None
    ipca = None
    fd = None

    global_step = 0
    for epoch in range(train_cfg["epochs"]):
        model.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=train_cfg["mixed_precision"]):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()

            if train_cfg["grad_clip_norm"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip_norm"])

            if collector.should_collect(global_step) and is_rank0(rank):
                grad_map = collector.collect(model)
                for key, vec in grad_map.items():
                    gram = gram_map.setdefault(key, GramAccumulator())
                    gram.add(vec)
                    if vec.numel() > 0 and key == "all":
                        if oja is None:
                            dim = vec.numel()
                            sketch_dim = cfg["analysis"]["streaming"]["sketch_dim"]
                            oja = OjaTracker(OjaConfig(components=sketch_dim, lr=0.01), dim)
                            ipca = IncrementalPCA(
                                IncrementalPCAConfig(components=sketch_dim, batch_size=16),
                                dim,
                            )
                            fd = FrequentDirections(sketch_dim, dim)
                        if cfg["analysis"]["streaming"]["oja"] and oja is not None:
                            oja.update(vec)
                        if cfg["analysis"]["streaming"]["incremental_pca"] and ipca is not None:
                            ipca.partial_fit(vec)
                        if cfg["analysis"]["streaming"]["frequent_directions"] and fd is not None:
                            fd.update(vec)

                if "all" in grad_map:
                    k_matrix = gram_map["all"].get_matrix().cpu().numpy()
                    eigenvalues = eigenvalues_from_gram(k_matrix)
                    summary = compute_summary(
                        eigenvalues,
                        cfg["analysis"]["spectral"]["energy_thresholds"],
                        cfg["analysis"]["spectral"]["powerlaw_fit"],
                    )
                    eff_rank_series.append(summary.effective_rank)

                for key, gram in gram_map.items():
                    if key == "all":
                        continue
                    k_matrix = gram.get_matrix().cpu().numpy()
                    eigenvalues = eigenvalues_from_gram(k_matrix)
                    summary = compute_summary(
                        eigenvalues,
                        cfg["analysis"]["spectral"]["energy_thresholds"],
                        cfg["analysis"]["spectral"]["powerlaw_fit"],
                    )
                    layer_rank_series.setdefault(key, []).append(summary.effective_rank)

            scaler.step(optimizer)
            scaler.update()

            if global_step % train_cfg["log_interval"] == 0 and is_rank0(rank):
                logger.info(
                    "epoch=%d step=%d loss=%.4f", epoch, global_step, loss.item()
                )
            global_step += 1

    if not is_rank0(rank):
        return

    if "all" in gram_map:
        K = gram_map["all"].get_matrix().cpu().numpy()
        np.save(output_dir / "spectra" / "gram_all.npy", K)
        projections = gram_map["all"].get_vectors().cpu().numpy()
        np.save(output_dir / "projections" / "gradients_all.npy", projections)
        eigenvalues = eigenvalues_from_gram(K)
        summary = compute_summary(
            eigenvalues,
            cfg["analysis"]["spectral"]["energy_thresholds"],
            cfg["analysis"]["spectral"]["powerlaw_fit"],
        )
        np.save(output_dir / "spectra" / "eigenvalues_all.npy", summary.eigenvalues)
        write_json(output_dir / "metrics" / "spectral_all.json", summary.__dict__)
        if cfg["visualization"]["enabled"]:
            set_style(cfg["visualization"]["style"])
            plot_scree(summary.eigenvalues, output_dir / "plots" / "scree_all.png")
            plot_log_decay(summary.eigenvalues, output_dir / "plots" / "log_decay_all.png")
            plot_cumulative_energy(
                summary.eigenvalues, output_dir / "plots" / "cumulative_all.png"
            )
            plot_effective_rank_over_time(
                eff_rank_series, output_dir / "plots" / "rank_over_time.png"
            )

    for key, gram in gram_map.items():
        if key == "all":
            continue
        safe_key = key.replace(".", "_")
        K = gram.get_matrix().cpu().numpy()
        np.save(output_dir / "spectra" / f"gram_{safe_key}.npy", K)
        projections = gram.get_vectors().cpu().numpy()
        np.save(output_dir / "projections" / f"gradients_{safe_key}.npy", projections)
        eigenvalues = eigenvalues_from_gram(K)
        summary = compute_summary(
            eigenvalues,
            cfg["analysis"]["spectral"]["energy_thresholds"],
            cfg["analysis"]["spectral"]["powerlaw_fit"],
        )
        np.save(output_dir / "spectra" / f"eigenvalues_{safe_key}.npy", summary.eigenvalues)
        write_json(output_dir / "metrics" / f"spectral_{safe_key}.json", summary.__dict__)

    if layer_rank_series and cfg["visualization"]["enabled"]:
        plot_layer_heatmap(layer_rank_series, output_dir / "plots" / "layer_rank.png")

    isotropy = {}
    for name, param in model.named_parameters():
        if param.ndim >= 2:
            isotropy[name] = isotropy_metrics(param)
    write_json(output_dir / "metrics" / "isotropy.json", isotropy)

    if oja is not None:
        np.save(output_dir / "metrics" / "oja_components.npy", oja.components.cpu().numpy())
    if ipca is not None and ipca.components is not None:
        np.save(output_dir / "metrics" / "ipca_components.npy", ipca.components.cpu().numpy())
    if fd is not None:
        fd_components = fd.get_components(cfg["analysis"]["streaming"]["sketch_dim"]).cpu().numpy()
        np.save(output_dir / "metrics" / "fd_components.npy", fd_components)

    logger.info("Training and analysis complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_loop(cfg)


if __name__ == "__main__":
    main()
