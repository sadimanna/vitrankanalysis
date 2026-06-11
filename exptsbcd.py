"""
gradient_manifold_analysis.py

Experiments:
B: Per-image gradient manifold intrinsic dimension
C: Gradient matrix rank / effective rank analysis
D: Jacobian spectrum and local recoverability analysis

Supports:
- Full fine-tuning
- LoRA (if model exposes requires_grad correctly)
- CIFAR10 / CIFAR100 / ImageNet100
- ViT-style models

This is a research scaffold intended for adaptation to your training codebase.
"""

import json
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from pathlib import Path
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from utils.projection import ProjectionConfig, project_vector

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14


# =========================
# ID ESTIMATORS
# =========================

def _fit_neighbors(X, k, chunk_size=8):
    """Memory-efficient k-NN distances using a chunked, row-by-row NumPy
    implementation. Returns an (N, k) array of distances to the k nearest
    neighbors of each row, excluding the self-match.

    Avoids sklearn's pairwise-distance "middle term" buffer that allocates
    an N*N*D float32 tensor (causes MemoryError on large gradient matrices).
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = X.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return np.empty((n, 0), dtype=np.float32)

    # Pre-normalize so ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    sq = (X * X).sum(axis=1)
    out = np.full((n, k), np.inf, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        a = X[start:end]
        a_sq = sq[start:end][:, None]
        # Batched squared distances against all rows
        d2 = a_sq + sq[None, :] - 2.0 * a.dot(X.T)
        np.maximum(d2, 0, out=d2)
        # For each query, take k+1 smallest, drop the self-match
        part = np.argpartition(d2, kth=k, axis=1)[:, :k]
        rows = np.arange(end - start)[:, None]
        # argpartition is unsorted within the k; sort those k indices by distance
        d2_part = np.take_along_axis(d2, part, axis=1)
        order = np.argsort(d2_part, axis=1)
        part_sorted = np.take_along_axis(part, order, axis=1)
        # Drop the self-match (distance 0) by shifting; safe because k <= n-1
        d_sorted = np.sqrt(np.take_along_axis(d2, part_sorted, axis=1))[:, 1:k + 1]
        out[start:end, :d_sorted.shape[1]] = d_sorted.astype(np.float32, copy=False)

    return out

def mle_id(X, n_neighbors=20):
    # MLE needs k+1 neighbors but k must be < N
    n_neighbors = min(n_neighbors, max(1, X.shape[0] - 1))
    d = _fit_neighbors(X, n_neighbors)
    if d.shape[1] < 2:
        return float("nan")
    ref = d[:, -1][:, None]
    ratios = ref / np.maximum(d[:, :-1], 1e-12)
    ids = 1.0 / np.maximum(np.mean(np.log(np.maximum(ratios, 1e-12)), axis=1), 1e-12)
    ids = ids[np.isfinite(ids)]
    return float(np.mean(ids)) if ids.size else float("nan")

def twonn_id(X, discard_fraction=0.1):
    d = _fit_neighbors(X, 2)
    r1, r2 = d[:, 0], d[:, 1]
    mu = np.sort(r2 / np.maximum(r1, 1e-12))
    n_keep = int(len(mu) * (1 - discard_fraction))
    mu = mu[:n_keep]
    f = np.arange(1, len(mu)+1, dtype=np.float64) / len(d)
    x = np.log(np.maximum(mu, 1e-12))
    y = -np.log(np.maximum(1 - f, 1e-12))
    return float(np.sum(x*y) / np.sum(x*x))

def l2n2_id(X, k=2, j=1):
    d = _fit_neighbors(X, max(k,j))
    rk = d[:, k-1]
    rj = d[:, j-1]
    ratio = rk / np.maximum(rj, 1e-12)
    vals = -np.log(np.log(np.maximum(ratio, 1+1e-12)))
    vals = vals[np.isfinite(vals)]
    return float(np.exp(vals.mean() + 0.57721))


# =========================
# EXPERIMENT B
# =========================

def collect_per_image_gradients(model, loader, device, proj_cfg=None, dtype=torch.float32):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]

    grads = []

    for x, y in tqdm(loader, desc="Collecting gradients"):
        x = x.to(device)
        y = y.to(device)

        for i in range(x.shape[0]):
            model.zero_grad(set_to_none=True)

            logits = model(x[i:i+1])
            loss = torch.nn.functional.cross_entropy(logits, y[i:i+1])

            g = torch.autograd.grad(
                loss,
                params,
                retain_graph=False,
                create_graph=False,
            )

            vec = torch.cat([v.detach().flatten() for v in g])
            if proj_cfg is not None and proj_cfg.enabled:
                vec = project_vector(vec, proj_cfg)
            grads.append(vec.detach().to(dtype=dtype).cpu().numpy())

    return np.stack(grads)


# =========================
# EXPERIMENT C
# =========================

def effective_rank_from_singular_values(S):
    p = S / (S.sum() + 1e-12)
    return float(np.exp(-(p * np.log(p + 1e-12)).sum()))

def analyze_gradient_matrix(G):
    Gc = G - G.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(Gc, full_matrices=False)

    energy = np.cumsum(S**2)
    energy /= energy[-1]

    rank90 = int(np.searchsorted(energy, 0.90) + 1)
    rank95 = int(np.searchsorted(energy, 0.95) + 1)
    rank99 = int(np.searchsorted(energy, 0.99) + 1)

    return {
        "effective_rank": effective_rank_from_singular_values(S),
        "rank90": rank90,
        "rank95": rank95,
        "rank99": rank99,
        "singular_values": S,
        "energy": energy,
    }

def plot_spectrum(S, out):
    plt.figure(figsize=(6,4))
    plt.semilogy(S)
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.title("Gradient Spectrum")
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    plt.close()

def plot_energy(E, out):
    plt.figure(figsize=(6,4))
    plt.plot(E)
    plt.xlabel("Rank")
    plt.ylabel("Cumulative Energy")
    plt.title("Gradient Energy")
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    plt.close()


# =========================
# EXPERIMENT D
# =========================

def jacobian_rank_analysis(model, image, label, device):

    params = [p for p in model.parameters() if p.requires_grad]

    image = image.unsqueeze(0).to(device)
    image.requires_grad_(True)

    label = torch.tensor([label], device=device)

    def grad_map(inp):
        logits = model(inp)
        loss = torch.nn.functional.cross_entropy(logits, label)

        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
        )

        return torch.cat([g.flatten() for g in grads])

    J = torch.autograd.functional.jacobian(
        grad_map,
        image,
        vectorize=True,
    )

    J = J.reshape(J.shape[0], -1).detach().cpu().numpy()

    U, S, Vt = np.linalg.svd(J, full_matrices=False)

    return {
        "jacobian_rank": int((S > 1e-8).sum()),
        "jacobian_effective_rank": effective_rank_from_singular_values(S),
        "singular_values": S,
    }

def plot_jacobian_spectrum(S, out):
    plt.figure(figsize=(6,4))
    plt.semilogy(S)
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.title("Jacobian Spectrum")
    plt.tight_layout()
    plt.savefig(out, dpi=600)
    plt.close()


# =========================
# RESULTS DRIVER
# =========================

def run_all(model, dataset, output_dir="results", batch_size=32, max_samples=100, proj_cfg=None):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    dataset = Subset(dataset, range(min(max_samples, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print("Collecting gradients...")
    G = collect_per_image_gradients(model, loader, device, proj_cfg=proj_cfg)

    print(G.shape)

    print("Experiment B")
    ids = {
        "MLE": mle_id(G),
        "L2N2": l2n2_id(G),
        "TwoNN": twonn_id(G),
    }

    print(ids)

    # print("Experiment C")
    # rank_stats = analyze_gradient_matrix(G)

    # plot_spectrum(
    #     rank_stats["singular_values"],
    #     output_dir / "gradient_spectrum.png",
    # )

    # plot_energy(
    #     rank_stats["energy"],
    #     output_dir / "gradient_energy.png",
    # )

    # print("Experiment D")
    # x, y = dataset[0]

    # jac = jacobian_rank_analysis(
    #     model,
    #     x,
    #     y,
    #     device,
    # )

    # plot_jacobian_spectrum(
    #     jac["singular_values"],
    #     output_dir / "jacobian_spectrum.png",
    # )

    pd.DataFrame([ids]).to_csv(
        output_dir / "gradient_ids.csv",
        index=False,
    )

    # pd.DataFrame([{
    #     k:v for k,v in rank_stats.items()
    #     if not isinstance(v, np.ndarray)
    # }]).to_csv(
    #     output_dir / "rank_metrics.csv",
    #     index=False,
    # )

    # pd.DataFrame([{
    #     k:v for k,v in jac.items()
    #     if not isinstance(v, np.ndarray)
    # }]).to_csv(
    #     output_dir / "jacobian_metrics.csv",
    #     index=False,
    # )

    # with open(output_dir / "summary.json", "w") as f:
    #     json.dump({
    #         "ids": ids,
    #         "rank_metrics": {
    #             k:v for k,v in rank_stats.items()
    #             if not isinstance(v, np.ndarray)
    #         },
    #         "jacobian_metrics": {
    #             k:v for k,v in jac.items()
    #             if not isinstance(v, np.ndarray)
    #         }
    #     }, f, indent=2)

    print("Finished.")

if __name__ == '__main__':
    # Example usage with a ViT-Small model (timm) and CIFAR10 dataset
    import timm

    model = timm.create_model("vit_small_patch16_224", pretrained=True, patch_size=16, img_size=32, num_classes=10)
    data_config = timm.data.resolve_model_data_config(model)
    # Force input size to match the model (timm's default data_config expects 224)
    data_config["input_size"] = (3, 32, 32)
    transform = timm.data.create_transform(**data_config, is_training=False)

    dataset = CIFAR10(root="data", train=True, download=True, transform=transform)

    # Random projection: matches the approach used in train.py / utils/projection.py
    proj_cfg = ProjectionConfig(
        enabled=True,
        method="gaussian",
        dim=8192,
        chunk_size=8192,  # No chunking for projection (fits in memory)
        seed=123,
    )

    run_all(model, dataset, output_dir="results/vit_cifar10", max_samples=1000, proj_cfg=proj_cfg)