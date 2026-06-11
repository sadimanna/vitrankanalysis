import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import torchvision

from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Data Loading
# ============================================================

def load_dataset(dataset_name="cifar10",
                 max_samples=5000):

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])

    if dataset_name.lower() == "cifar10":
        dataset = CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )

    elif dataset_name.lower() == "cifar100":
        dataset = CIFAR100(
            root="./data",
            train=True,
            download=True,
            transform=transform
        )
    elif dataset_name.lower() == "imagenet100":
        dataset = torchvision.datasets.ImageFolder(
            root="/home/siladittyamanna/Documents/smanna/iisc/work1/dataset/imagenet100/train",
            transform=transform
        )
    else:
        raise ValueError(
            "dataset_name must be cifar10 or cifar100"
        )

    idx = np.random.choice(
        len(dataset),
        size=min(max_samples, len(dataset)),
        replace=False
    )

    images = []

    for i in idx:
        img, _ = dataset[i]
        images.append(
            img.permute(1, 2, 0).numpy()
        )

    images = np.stack(images)

    return images


# ============================================================
# Preprocessing
# ============================================================

def flatten_images(images):

    X = images.reshape(
        len(images),
        -1
    ).astype(np.float32)

    X -= X.mean(axis=0)

    return X


def _fit_neighbors(X, n_neighbors):
    n_neighbors = min(int(n_neighbors), len(X) - 1)
    if n_neighbors < 2:
        raise ValueError("Need at least 3 samples to estimate intrinsic dimension")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto")
    distances, _ = nbrs.fit(X).kneighbors(X)
    return distances[:, 1:]


# ============================================================
# MLE Intrinsic Dimension
# ============================================================

def compute_mle_dimension(
    X,
    n_neighbors=20
):

    neighbor_distances = _fit_neighbors(X, n_neighbors)
    reference_distance = neighbor_distances[:, -1][:, None]
    ratios = reference_distance / np.maximum(neighbor_distances[:, :-1], 1e-12)
    local_ids = 1.0 / np.maximum(np.mean(np.log(np.maximum(ratios, 1e-12)), axis=1), 1e-12)
    local_ids = local_ids[np.isfinite(local_ids)]
    return float(np.mean(local_ids)) if local_ids.size else None


def compute_local_mle_dimension(
    X,
    n_neighbors=20
):

    neighbor_distances = _fit_neighbors(X, n_neighbors)
    reference_distance = neighbor_distances[:, -1][:, None]
    ratios = reference_distance / np.maximum(neighbor_distances[:, :-1], 1e-12)
    local_ids = 1.0 / np.maximum(np.mean(np.log(np.maximum(ratios, 1e-12)), axis=1), 1e-12)
    return local_ids[np.isfinite(local_ids)]


def compute_l2n2_dimension(
    X,
    k=2,
    j=1,
    alpha=None,
    beta=None,
):

    if alpha is None or beta is None:
        alpha = 1.0
        beta = 0.57721

    neighbor_distances = _fit_neighbors(X, max(k, j))
    rk = neighbor_distances[:, k - 1]
    rj = neighbor_distances[:, j - 1]
    ratio = rk / np.maximum(rj, 1e-12)
    lkj_values = -np.log(np.log(np.maximum(ratio, 1.0 + 1e-12)))
    lkj_values = lkj_values[np.isfinite(lkj_values)]
    if lkj_values.size == 0:
        return None
    l_bar = float(np.mean(lkj_values))
    return float(np.exp(alpha * l_bar + beta))


def compute_twonn_dimension(
    X,
    discard_fraction=0.1,
):

    neighbor_distances = _fit_neighbors(X, 2)
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


# ============================================================
# Isomap Dimension Estimation
# ============================================================

def compute_isomap_curve(
    X,
    max_dim=50,
    n_neighbors=10
):

    dims = np.arange(
        1,
        max_dim + 1
    )

    errors = []

    for d in dims:

        print(
            f"Running Isomap for dimension {d}"
        )

        iso = Isomap(
            n_neighbors=n_neighbors,
            n_components=d
        )

        iso.fit(X)

        errors.append(
            iso.reconstruction_error()
        )

    return dims, np.array(errors)


def estimate_elbow(errors):

    second_derivative = np.diff(
        errors,
        n=2
    )

    elbow = (
        np.argmax(
            np.abs(second_derivative)
        ) + 2
    )

    return elbow


# ============================================================
# Plotting
# ============================================================

def plot_local_id_histogram(
    local_ids,
    save_path=None
):

    plt.figure(figsize=(8, 5))

    plt.hist(
        local_ids,
        bins=40
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

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close()


def plot_isomap_curve(
    dims,
    errors,
    elbow,
    save_path=None
):

    plt.figure(figsize=(8, 5))

    plt.plot(
        dims,
        errors,
        marker='o'
    )

    plt.axvline(
        elbow,
        linestyle='--',
        label=f'Elbow ≈ {elbow}'
    )

    plt.xlabel(
        "Embedding Dimension"
    )

    plt.ylabel(
        "Reconstruction Error"
    )

    plt.title(
        "Isomap Dimension Estimate"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close()


def plot_intrinsic_dimension_sweep(
    sample_sizes,
    estimates,
    save_path=None,
):

    plt.figure(figsize=(8, 5))

    for label, values in estimates.items():
        plt.plot(sample_sizes, values, marker="o", label=label)

    plt.xlabel("Number of samples")
    plt.ylabel("Estimated intrinsic dimension")
    plt.title("Intrinsic Dimension Estimates vs Sample Size")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close()


def plot_intrinsic_dimension_trials(
    sample_sizes,
    trial_estimates,
    save_path=None,
):

    plt.figure(figsize=(9, 5.5))

    for label, values in trial_estimates.items():
        values = np.asarray(values, dtype=np.float64)

        for trial_values in values:
            plt.plot(
                sample_sizes,
                trial_values,
                marker="o",
                linewidth=1.0,
                alpha=0.18,
            )

        mean_values = np.nanmean(values, axis=0)
        std_values = np.nanstd(values, axis=0)

        plt.plot(
            sample_sizes,
            mean_values,
            marker="o",
            linewidth=2.5,
            label=f"{label} mean",
        )

        plt.fill_between(
            sample_sizes,
            mean_values - std_values,
            mean_values + std_values,
            alpha=0.15,
        )

    plt.xlabel("Number of samples")
    plt.ylabel("Estimated intrinsic dimension")
    plt.title("Intrinsic Dimension Estimates Across Random Trials")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    DATASET = "imagenet100"

    MAX_SAMPLES = 50000

    SAMPLE_SIZES = [50000]
    N_TRIALS = 1
    TRIAL_SEED = 42

    ISOMAP_MAX_DIM = 50

    ISOMAP_NEIGHBORS = 10

    MLE_NEIGHBORS = 100
    L2N2_K = 2
    L2N2_J = 1
    TWONN_DISCARD_FRACTION = 0.1


    # --------------------------------------------------------

    print("Loading dataset...")

    images = load_dataset(
        DATASET,
        max_samples=MAX_SAMPLES
    )

    print(
        "Images shape:",
        images.shape
    )

    X = flatten_images(images)

    print(
        "Flattened shape:",
        X.shape
    )

    rng = np.random.default_rng(TRIAL_SEED)

    trial_estimates = {
        "MLE": [],
        "L2N2": [],
        "TWO-NN": [],
    }

    for trial_idx in range(N_TRIALS):
        order = rng.permutation(len(X))
        X_trial = X[order]

        mle_values = []
        l2n2_values = []
        twonn_values = []

        print(f"\nStarting trial {trial_idx + 1}/{N_TRIALS}...")

        for sample_size in SAMPLE_SIZES:
            X_subset = X_trial[:sample_size]
            print(f"Computing intrinsic dimensions for {sample_size} samples...")

            mle_id = compute_mle_dimension(X_subset, n_neighbors=MLE_NEIGHBORS)
            l2n2_id = compute_l2n2_dimension(
                X_subset,
                k=L2N2_K,
                j=L2N2_J,
            )
            twonn_id = compute_twonn_dimension(
                X_subset,
                discard_fraction=TWONN_DISCARD_FRACTION,
            )

            mle_values.append(np.nan if mle_id is None else float(mle_id))
            l2n2_values.append(np.nan if l2n2_id is None else float(l2n2_id))
            twonn_values.append(np.nan if twonn_id is None else float(twonn_id))

            print(f"MLE   = {mle_values[-1]:.2f}")
            print(f"L2N2  = {l2n2_values[-1]:.2f}")
            print(f"TWO-NN = {twonn_values[-1]:.2f}")

        trial_estimates["MLE"].append(mle_values)
        trial_estimates["L2N2"].append(l2n2_values)
        trial_estimates["TWO-NN"].append(twonn_values)

    plot_intrinsic_dimension_trials(
        SAMPLE_SIZES,
        trial_estimates,
        save_path=f"{DATASET}_intrinsic_dimension_trials.png",
    )

    plot_intrinsic_dimension_sweep(
        SAMPLE_SIZES,
        {
            label: np.nanmean(np.asarray(values, dtype=np.float64), axis=0).tolist()
            for label, values in trial_estimates.items()
        },
        save_path=f"{DATASET}_intrinsic_dimension_trials_mean.png",
    )

    local_ids = compute_local_mle_dimension(
        X_trial[:SAMPLE_SIZES[-1]],
        n_neighbors=MLE_NEIGHBORS
    )

    plot_local_id_histogram(
        local_ids,
        save_path=f"{DATASET}_local_id_hist.png"
    )

    # --------------------------------------------------------
    # Isomap
    # --------------------------------------------------------

    # print(
    #     "\nComputing Isomap curve..."
    # )

    # dims, errors = compute_isomap_curve(
    #     X,
    #     max_dim=ISOMAP_MAX_DIM,
    #     n_neighbors=ISOMAP_NEIGHBORS
    # )

    # elbow = estimate_elbow(errors)

    # print(
    #     f"Estimated Isomap Dimension = {elbow}"
    # )

    # plot_isomap_curve(
    #     dims,
    #     errors,
    #     elbow,
    #     save_path=f"{DATASET}_isomap_curve.png"
    # )

    print("\nDone.")