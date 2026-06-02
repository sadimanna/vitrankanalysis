"""Training entry point."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
import torch
try:
    from torch.amp import GradScaler as AmpGradScaler
    from torch.amp import autocast as amp_autocast
except Exception:  # pragma: no cover - older torch
    AmpGradScaler = None
    amp_autocast = None
from torch.cuda.amp import GradScaler as CudaGradScaler
from torch.cuda.amp import autocast as cuda_autocast

from analysis.gram import GramAccumulator
from analysis.gradient_collector import GradientCollectConfig, GradientCollector
from analysis.isotropy import isotropy_metrics
from analysis.spectral import compute_summary, eigenvalues_from_gram
from analysis.streaming import FrequentDirections, IncrementalPCA, IncrementalPCAConfig, OjaConfig, OjaTracker
from models.lora import LoRAConfig, lora_parameters
from models.vit_factory import create_vit
from training.data import get_dataloaders
from training.optim import build_optimizer, build_scheduler
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


def _evaluate_accuracy(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    autocast_ctx = amp_autocast if amp_autocast is not None else cuda_autocast
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            with autocast_ctx(
                **({"device_type": device.type} if autocast_ctx is amp_autocast else {}),
                enabled=use_amp,
            ):
                logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.numel()
    model.train()
    return float(correct) / max(float(total), 1.0)


def _compute_mle_intrinsic_dimension(
    samples: np.ndarray,
    n_neighbors: int,
) -> float | None:
    if samples.shape[0] < 3:
        return None

    k = min(int(n_neighbors), samples.shape[0] - 1)
    if k < 2:
        return None

    diffs = samples[:, None, :] - samples[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=-1))
    np.fill_diagonal(distances, np.inf)
    neighbor_distances = np.sort(distances, axis=1)[:, :k]
    reference_distance = neighbor_distances[:, -1][:, None]
    ratios = reference_distance / np.maximum(neighbor_distances[:, :-1], 1e-12)
    local_denominator = np.mean(np.log(np.maximum(ratios, 1e-12)), axis=1)
    local_ids = 1.0 / np.maximum(local_denominator, 1e-12)
    local_ids = local_ids[np.isfinite(local_ids)]
    if local_ids.size == 0:
        return None
    return float(np.mean(local_ids))


def _pairwise_distances(samples: np.ndarray) -> np.ndarray:
    diffs = samples[:, None, :] - samples[None, :, :]
    distances = np.sqrt(np.sum(diffs * diffs, axis=-1))
    np.fill_diagonal(distances, np.inf)
    return distances


def _compute_l2n2_intrinsic_dimension(
    samples: np.ndarray,
    k: int,
    j: int,
    alpha: float | None = None,
    beta: float | None = None,
) -> float | None:
    if samples.shape[0] < max(k, j) + 1:
        return None

    if alpha is None or beta is None:
        alpha = 1.0
        beta = 0.57721

    distances = _pairwise_distances(samples)
    neighbor_distances = np.sort(distances, axis=1)
    rk = neighbor_distances[:, k - 1]
    rj = neighbor_distances[:, j - 1]
    ratio = rk / np.maximum(rj, 1e-12)
    lkj_values = -np.log(np.log(np.maximum(ratio, 1.0 + 1e-12)))
    lkj_values = lkj_values[np.isfinite(lkj_values)]
    if lkj_values.size == 0:
        return None

    l_bar = float(np.mean(lkj_values))
    return float(np.exp(alpha * l_bar + beta))


def _compute_twonn_intrinsic_dimension(
    samples: np.ndarray,
    discard_fraction: float,
) -> float | None:
    if samples.shape[0] < 3:
        return None

    distances = _pairwise_distances(samples)
    neighbor_distances = np.sort(distances, axis=1)
    r1 = neighbor_distances[:, 0]
    r2 = neighbor_distances[:, 1]
    mu = r2 / np.maximum(r1, 1e-12)
    mu_sorted = np.sort(mu)
    n_keep = int(len(mu_sorted) * (1 - discard_fraction))
    if n_keep < 2:
        return None

    mu_final = mu_sorted[:n_keep]
    f_final = np.arange(1, len(mu_sorted) + 1, dtype=np.float64)[:n_keep] / len(mu_sorted)
    x = np.log(np.maximum(mu_final, 1e-12))
    y = -np.log(np.maximum(1 - f_final, 1e-12))
    denom = float(np.sum(x * x))
    if denom <= 0:
        return None
    return float(np.sum(x * y) / denom)


def train_loop(cfg: Dict) -> None:
    distributed, rank = init_distributed()
    set_seed(cfg["seed"], deterministic=False)

    output_root = Path(cfg["output_dir"])
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = output_root / run_name
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

    train_loader, test_loader, num_classes = get_dataloaders(
        data_cfg["name"],
        data_cfg["data_dir"],
        data_cfg["image_size"],
        train_cfg["batch_size"],
        data_cfg["num_workers"],
        augmentation=data_cfg.get("augmentation"),
    )

    lora_cfg = LoRAConfig(**cfg["model"]["lora"])
    model = create_vit(
        cfg["model"]["source"],
        cfg["model"]["name"],
        cfg["model"]["pretrained"],
        num_classes,
        lora_cfg,
        image_size=data_cfg["image_size"],
        patch_size=data_cfg["patch_size"],
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
    scheduler_cfg = train_cfg.get("scheduler", {"name": "none"})
    scheduler = build_scheduler(optimizer, scheduler_cfg, train_cfg["epochs"])

    if AmpGradScaler is not None:
        try:
            scaler = AmpGradScaler(
                device_type=device.type, enabled=train_cfg["mixed_precision"]
            )
        except TypeError:
            try:
                scaler = AmpGradScaler(device.type, enabled=train_cfg["mixed_precision"])
            except TypeError:
                scaler = CudaGradScaler(enabled=train_cfg["mixed_precision"])
    else:
        scaler = CudaGradScaler(enabled=train_cfg["mixed_precision"])
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
    test_acc_series: List[float] = []

    for epoch in range(train_cfg["epochs"]):
        model.train()
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)
            with (amp_autocast if amp_autocast is not None else cuda_autocast)(
                **({"device_type": device.type} if amp_autocast is not None else {}),
                enabled=train_cfg["mixed_precision"],
            ):
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

        if is_rank0(rank):
            test_acc = _evaluate_accuracy(
                model, test_loader, device, train_cfg["mixed_precision"]
            )
            test_acc_series.append(test_acc)
            logger.info("epoch=%d test_acc=%.4f", epoch, test_acc)

        if scheduler is not None:
            scheduler.step()

    if not is_rank0(rank):
        return

    if "all" in gram_map:
        K = gram_map["all"].get_matrix().cpu().numpy()
        np.save(output_dir / "spectra" / "gram_all.npy", K)
        projections = gram_map["all"].get_vectors().cpu().numpy()
        np.save(output_dir / "projections" / "gradients_all.npy", projections)
        mle_cfg = cfg["analysis"].get("mle", {})
        l2n2_cfg = cfg["analysis"].get("l2n2", {})
        twonn_cfg = cfg["analysis"].get("twonn", {})

        mle_neighbors = int(mle_cfg.get("n_neighbors", 20))
        mle_id = _compute_mle_intrinsic_dimension(projections, mle_neighbors)
        if mle_id is not None:
            write_json(
                output_dir / "metrics" / "mle_all.json",
                {
                    "value": mle_id,
                    "n_neighbors": min(mle_neighbors, max(projections.shape[0] - 1, 0)),
                    "num_samples": int(projections.shape[0]),
                },
            )
            logger.info("MLE intrinsic dimension = %.4f", mle_id)
        else:
            logger.warning("Skipping MLE intrinsic dimension: insufficient gradient samples")

        l2n2_k = int(l2n2_cfg.get("k", 2))
        l2n2_j = int(l2n2_cfg.get("j", 1))
        l2n2_alpha = l2n2_cfg.get("alpha")
        l2n2_beta = l2n2_cfg.get("beta")
        l2n2_id = _compute_l2n2_intrinsic_dimension(
            projections,
            l2n2_k,
            l2n2_j,
            l2n2_alpha,
            l2n2_beta,
        )
        if l2n2_id is not None:
            write_json(
                output_dir / "metrics" / "l2n2_all.json",
                {
                    "value": l2n2_id,
                    "k": l2n2_k,
                    "j": l2n2_j,
                    "num_samples": int(projections.shape[0]),
                },
            )
            logger.info("L2N2 intrinsic dimension = %.4f", l2n2_id)
        else:
            logger.warning("Skipping L2N2 intrinsic dimension: insufficient gradient samples")

        twonn_discard_fraction = float(twonn_cfg.get("discard_fraction", 0.1))
        twonn_id = _compute_twonn_intrinsic_dimension(projections, twonn_discard_fraction)
        if twonn_id is not None:
            write_json(
                output_dir / "metrics" / "twonn_all.json",
                {
                    "value": twonn_id,
                    "discard_fraction": twonn_discard_fraction,
                    "num_samples": int(projections.shape[0]),
                },
            )
            logger.info("TWO-NN intrinsic dimension = %.4f", twonn_id)
        else:
            logger.warning("Skipping TWO-NN intrinsic dimension: insufficient gradient samples")
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
    if test_acc_series:
        write_json(
            output_dir / "metrics" / "test_accuracy.json",
            {"values": test_acc_series},
        )

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
