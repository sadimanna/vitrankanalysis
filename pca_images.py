import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms


def plot_pca_spectrum(images,
                      max_components=None,
                      title="PCA Eigenvalue Spectrum",
                      save_path=None):
    """
    images: numpy array of shape (N,H,W,C)
    """

    N = images.shape[0]

    # Flatten images
    X = images.reshape(N, -1).astype(np.float32)

    # Center data
    X -= X.mean(axis=0)

    if max_components is None:
        max_components = min(N, X.shape[1])

    print(f"Running PCA on {X.shape}, max_components={max_components}")

    pca = PCA(
        n_components=max_components,
        svd_solver="randomized",
        random_state=42
    )

    pca.fit(X)

    eigvals = pca.explained_variance_
    print(f"Number of eigvals: {len(eigvals)}")

    plt.figure(figsize=(8, 5))
    plt.plot(
        np.arange(1, len(eigvals) + 1),
        eigvals
    )

    plt.yscale("log")
    plt.xlabel("Principal Component")
    plt.ylabel("Eigenvalue (log scale)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved PCA spectrum plot to {save_path}")

    plt.close()

    return eigvals

if __name__=="__main__":
    dataset = CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    images = np.stack([
        dataset[i][0].permute(1, 2, 0).numpy()
        for i in range(len(dataset))
    ])

    eigvals = plot_pca_spectrum(
        images,
        max_components=3072,
        title="CIFAR-10 PCA Spectrum",
        save_path="outputs/cifar10_pca_spectrum.png"
    )


    # def get_pca_spectrum(images, n_components=1000):
    #     X = images.reshape(len(images), -1).astype(np.float32)
    #     X -= X.mean(axis=0)

    #     pca = PCA(
    #         n_components=n_components,
    #         svd_solver="randomized",
    #         random_state=42
    #     )

    #     pca.fit(X)
    #     return pca.explained_variance_


    # spec_c10 = get_pca_spectrum(cifar10_images)
    # spec_c100 = get_pca_spectrum(cifar100_images)
    # spec_mini = get_pca_spectrum(miniimagenet_images)

    # plt.figure(figsize=(8,5))

    # plt.plot(spec_c10, label="CIFAR10")
    # plt.plot(spec_c100, label="CIFAR100")
    # plt.plot(spec_mini, label="MiniImageNet100")

    # plt.yscale("log")
    # plt.xlabel("Principal Component")
    # plt.ylabel("Eigenvalue")
    # plt.title("PCA Eigenspectrum")
    # plt.legend()
    # plt.grid(True)

    # plt.show()