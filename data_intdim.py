import numpy as np
import torch
import timm
from tqdm import tqdm
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


# ============================================================
# CONFIG
# ============================================================

DATASET = "imagenet100"  # cifar10 | cifar100 | imagenet100

MAX_SAMPLES = 50000

BATCH_SIZE = 128

MODEL_NAME = "vit_small_patch16_224"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_PCA = True
PCA_DIM = 512

MLE_NEIGHBORS = 500

L2N2_K = 2
L2N2_J = 1

TWONN_DISCARD_FRACTION = 0.1

N_TRIALS = 1
TRIAL_SEED = 42

SAMPLE_SIZES = [MAX_SAMPLES]


# ============================================================
# DATASET LOADING
# ============================================================

def load_dataset(
    dataset_name="cifar10",
    max_samples=5000,
):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    if dataset_name.lower() == "cifar10":
        dataset = CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

    elif dataset_name.lower() == "cifar100":
        dataset = CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=transform,
        )

    elif dataset_name.lower() == "imagenet100":
        dataset = ImageFolder(
            root="/home/siladittyamanna/Documents/smanna/iisc/work1/dataset/imagenet100/train",
            transform=transform,
        )

    else:
        raise ValueError(dataset_name)

    idx = np.random.choice(
        len(dataset),
        size=min(max_samples, len(dataset)),
        replace=False,
    )

    dataset = Subset(dataset, idx)

    return dataset


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_vit_features(
    dataset,
    model_name="vit_base_patch16_224",
    batch_size=128,
    device="cuda",
):
    print(f"Loading model: {model_name}")

    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=0,
    )

    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    features = []

    with torch.no_grad():

        for images, _ in tqdm(loader):

            images = images.to(device)

            feat = model.forward_features(images)

            if isinstance(feat, tuple):
                feat = feat[0]

            if feat.ndim == 3:
                feat = feat[:, 0]

            features.append(
                feat.cpu().numpy()
            )

    features = np.concatenate(
        features,
        axis=0,
    )

    return features.astype(np.float32)


# ============================================================
# NEIGHBORS
# ============================================================

def _fit_neighbors(
    X,
    n_neighbors,
):
    n_neighbors = min(
        int(n_neighbors),
        len(X) - 1,
    )

    nbrs = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        algorithm="auto",
    )

    distances, _ = nbrs.fit(X).kneighbors(X)

    return distances[:, 1:]


# ============================================================
# MLE ID
# ============================================================

def compute_mle_dimension(
    X,
    n_neighbors=20,
):
    neighbor_distances = _fit_neighbors(
        X,
        n_neighbors,
    )

    reference_distance = (
        neighbor_distances[:, -1][:, None]
    )

    ratios = (
        reference_distance
        / np.maximum(
            neighbor_distances[:, :-1],
            1e-12,
        )
    )

    local_ids = (
        1.0
        / np.maximum(
            np.mean(
                np.log(
                    np.maximum(
                        ratios,
                        1e-12,
                    )
                ),
                axis=1,
            ),
            1e-12,
        )
    )

    local_ids = local_ids[np.isfinite(local_ids)]

    if local_ids.size == 0:
        return None

    return float(np.mean(local_ids))


# ============================================================
# LOCAL MLE
# ============================================================

def compute_local_mle_dimension(
    X,
    n_neighbors=20,
):
    neighbor_distances = _fit_neighbors(
        X,
        n_neighbors,
    )

    reference_distance = (
        neighbor_distances[:, -1][:, None]
    )

    ratios = (
        reference_distance
        / np.maximum(
            neighbor_distances[:, :-1],
            1e-12,
        )
    )

    local_ids = (
        1.0
        / np.maximum(
            np.mean(
                np.log(
                    np.maximum(
                        ratios,
                        1e-12,
                    )
                ),
                axis=1,
            ),
            1e-12,
        )
    )

    return local_ids[np.isfinite(local_ids)]


# ============================================================
# L2N2
# ============================================================

def compute_l2n2_dimension(
    X,
    k=2,
    j=1,
    alpha=None,
    beta=None,
):
    if alpha is None:
        alpha = 1.0

    if beta is None:
        beta = 0.57721

    neighbor_distances = _fit_neighbors(
        X,
        max(k, j),
    )

    rk = neighbor_distances[:, k - 1]

    rj = neighbor_distances[:, j - 1]

    ratio = rk / np.maximum(
        rj,
        1e-12,
    )

    lkj_values = -np.log(
        np.log(
            np.maximum(
                ratio,
                1.0 + 1e-12,
            )
        )
    )

    lkj_values = lkj_values[
        np.isfinite(lkj_values)
    ]

    if lkj_values.size == 0:
        return None

    l_bar = float(
        np.mean(
            lkj_values
        )
    )

    return float(
        np.exp(
            alpha * l_bar + beta
        )
    )


# ============================================================
# TWO NN
# ============================================================

def compute_twonn_dimension(
    X,
    discard_fraction=0.1,
):
    neighbor_distances = _fit_neighbors(
        X,
        2,
    )

    r1 = neighbor_distances[:, 0]

    r2 = neighbor_distances[:, 1]

    mu = r2 / np.maximum(
        r1,
        1e-12,
    )

    mu_sorted = np.sort(mu)

    n_keep = int(
        len(mu_sorted)
        * (1 - discard_fraction)
    )

    if n_keep < 2:
        return None

    mu_final = mu_sorted[:n_keep]

    f_final = (
        np.arange(
            1,
            len(mu_sorted) + 1,
            dtype=np.float64,
        )[:n_keep]
        / len(mu_sorted)
    )

    x = np.log(
        np.maximum(
            mu_final,
            1e-12,
        )
    )

    y = -np.log(
        np.maximum(
            1 - f_final,
            1e-12,
        )
    )

    denom = float(
        np.sum(
            x * x
        )
    )

    if denom <= 0:
        return None

    return float(
        np.sum(x * y)
        / denom
    )


# ============================================================
# HISTOGRAM
# ============================================================

def plot_local_id_histogram(
    local_ids,
    save_path,
):
    plt.figure(figsize=(8, 5))

    plt.hist(
        local_ids,
        bins=40,
    )

    plt.xlabel(
        "Local Intrinsic Dimension"
    )

    plt.ylabel(
        "Count"
    )

    plt.title(
        "MLE Local Intrinsic Dimensions"
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\nLoading dataset...")

    dataset = load_dataset(
        DATASET,
        max_samples=MAX_SAMPLES,
    )

    print(
        "Dataset size:",
        len(dataset),
    )

    print(
        "\nExtracting ViT features..."
    )

    X = extract_vit_features(
        dataset,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    print(
        "Feature matrix shape:",
        X.shape,
    )

    X -= X.mean(
        axis=0,
        keepdims=True,
    )

    if USE_PCA:

        print(
            f"Running PCA -> {PCA_DIM}"
        )

        pca = PCA(
            n_components=min(
                PCA_DIM,
                X.shape[1],
            ),
            whiten=False,
        )

        X = pca.fit_transform(X)

        print(
            "PCA shape:",
            X.shape,
        )

    rng = np.random.default_rng(
        TRIAL_SEED
    )

    for trial_idx in range(
        N_TRIALS
    ):

        print(
            f"\nTrial {trial_idx+1}/{N_TRIALS}"
        )

        order = rng.permutation(
            len(X)
        )

        X_trial = X[order]

        for sample_size in SAMPLE_SIZES:

            X_subset = X_trial[
                :sample_size
            ]

            print(
                f"\nSamples = {sample_size}"
            )

            mle_id = compute_mle_dimension(
                X_subset,
                n_neighbors=MLE_NEIGHBORS,
            )

            l2n2_id = compute_l2n2_dimension(
                X_subset,
                k=L2N2_K,
                j=L2N2_J,
            )

            twonn_id = compute_twonn_dimension(
                X_subset,
                discard_fraction=TWONN_DISCARD_FRACTION,
            )

            print(
                f"MLE    = {mle_id:.2f}"
            )

            print(
                f"L2N2   = {l2n2_id:.2f}"
            )

            print(
                f"TWO-NN = {twonn_id:.2f}"
            )

    local_ids = compute_local_mle_dimension(
        X,
        n_neighbors=MLE_NEIGHBORS,
    )

    plot_local_id_histogram(
        local_ids,
        save_path=f"{DATASET}_feature_local_id_hist.png",
    )

    print("\nDone.")