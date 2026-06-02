import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms

from sklearn.manifold import Isomap
from skdim.id import MLE


# ============================================================
# Data Loading
# ============================================================

def load_dataset(dataset_name="cifar10",
                 max_samples=5000):

    transform = transforms.ToTensor()

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


# ============================================================
# MLE Intrinsic Dimension
# ============================================================

def compute_mle_dimension(
    X,
    n_neighbors=20
):

    estimator = MLE()

    global_id = estimator.fit_transform(
        X,
        n_neighbors=n_neighbors
    )

    return global_id


def compute_local_mle_dimension(
    X,
    n_neighbors=20
):

    estimator = MLE()

    local_ids = estimator.fit_transform_pw(
        X,
        n_neighbors=n_neighbors
    )

    return local_ids


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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    DATASET = "cifar10"

    MAX_SAMPLES = 5000

    ISOMAP_MAX_DIM = 50

    ISOMAP_NEIGHBORS = 10

    MLE_NEIGHBORS = 20


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

    # --------------------------------------------------------
    # MLE Global ID
    # --------------------------------------------------------

    print(
        "\nComputing MLE intrinsic dimension..."
    )

    mle_id = compute_mle_dimension(
        X,
        n_neighbors=MLE_NEIGHBORS
    )

    print(
        f"MLE Intrinsic Dimension = {mle_id:.2f}"
    )

    # --------------------------------------------------------
    # Local IDs
    # --------------------------------------------------------

    print(
        "\nComputing local MLE dimensions..."
    )

    local_ids = compute_local_mle_dimension(
        X,
        n_neighbors=MLE_NEIGHBORS
    )

    plot_local_id_histogram(
        local_ids,
        save_path=f"{DATASET}_local_id_hist.png"
    )

    # --------------------------------------------------------
    # Isomap
    # --------------------------------------------------------

    print(
        "\nComputing Isomap curve..."
    )

    dims, errors = compute_isomap_curve(
        X,
        max_dim=ISOMAP_MAX_DIM,
        n_neighbors=ISOMAP_NEIGHBORS
    )

    elbow = estimate_elbow(errors)

    print(
        f"Estimated Isomap Dimension = {elbow}"
    )

    plot_isomap_curve(
        dims,
        errors,
        elbow,
        save_path=f"{DATASET}_isomap_curve.png"
    )

    print("\nDone.")