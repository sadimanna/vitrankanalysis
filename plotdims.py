import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# -----------------------------
# NeurIPS-style settings
# -----------------------------
plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# -----------------------------
# Helper: filter out NaN entries
# -----------------------------
def filter_nans(data, labels):
    """Remove NaN entries from data and the corresponding labels."""
    data = np.asarray(data, dtype=float)
    mask = ~np.isnan(data)
    return data[mask].tolist(), [l for l, m in zip(labels, mask) if m]

# -----------------------------
# Data
# -----------------------------
ranks = ["2", "8", "16", "32", "64", "128", "Full"]

# Projection dim = 1024
acc_1024 = [0.7870, 0.8099, 0.8438, 0.8491, 0.8558, 0.8591, 0.8394]

mle_1024 = [26.5576, 34.4445, 72.0084, 96.3921, 120.2322, 127.5194, 99.2650]
twonn_1024 = [33.2982, 38.4158, 82.6776, 106.3492, 135.0258, 146.8584, 109.4200]
l2n2_1024 = [99.6450, 115.7459, 258.1975, 332.4786, 434.2945, 484.9815, 349.3169]

# Projection dim = 8192
acc_8192 = [np.nan, np.nan, 0.8408, 0.8496, 0.8532, 0.8591, 0.8389]

mle_8192 = [np.nan, np.nan, 89.5516, 138.1012, 199.2572, 260.2628, 137.5167]
twonn_8192 = [np.nan, np.nan, 95.8630, 151.8194, 209.6191, 273.8720, 145.8027]
l2n2_8192 = [np.nan, np.nan, 314.3398, 493.5167, 669.0973, 906.0413, 437.5667]

# Dataset references
dataset_refs = {
    "CIFAR10": {
        "MLE": 27.16,
        "TWO-NN": 31.66,
        "L2N2": 101.28
    },
    "CIFAR100": {
        "MLE": 24.65,
        "TWO-NN": 28.43,
        "L2N2": 93.82
    },
    "ImageNet100 Train": {
        "MLE": 29.99,
        "TWO-NN": 40.89,
        "L2N2": 137.88
    },
    "ImageNet100 Val": {
        "MLE": 19.82,
        "TWO-NN": 31.89,
        "L2N2": 104.61
    }
}

# --------------------------------------------------
# Accuracy plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(7,4))

# Filter NaNs per series
acc_1024_f, ranks_1024 = filter_nans(acc_1024, ranks)
acc_8192_f, ranks_8192 = filter_nans(acc_8192, ranks)

x_1024 = np.arange(len(ranks_1024))
x_8192 = np.arange(len(ranks_8192))
width = 0.35

ax.bar(x_1024 - width/2, acc_1024_f, width,
       label='Proj Dim 1024')
ax.bar(x_8192 + width/2, acc_8192_f, width,
       label='Proj Dim 8192')

ax.set_ylabel("Test Accuracy")
ax.set_xlabel("LoRA Rank")
ax.set_title("Accuracy vs LoRA Rank")

# Use the union of available ranks for the x-axis ticks
all_ranks = ranks_1024 + [r for r in ranks_8192 if r not in ranks_1024]
ax.set_xticks(np.arange(len(all_ranks)))
ax.set_xticklabels(all_ranks)
ax.set_ylim(0.75, 0.88)
ax.legend(frameon=False)

plt.tight_layout()
plt.savefig("accuracy_vs_rank.pdf")
plt.savefig("accuracy_vs_rank.png")
plt.show()

# --------------------------------------------------
# Intrinsic dimension plot helper
# --------------------------------------------------
def plot_intrinsic_dimensions(
        mle,
        twonn,
        l2n2,
        proj_dim):

    fig, ax = plt.subplots(figsize=(8,4))

    # Filter NaNs per series
    mle_f, ranks_mle = filter_nans(mle, ranks)
    twonn_f, ranks_twonn = filter_nans(twonn, ranks)
    l2n2_f, ranks_l2n2 = filter_nans(l2n2, ranks)

    # Union of all available rank labels (preserves order from `ranks`)
    used_ranks = []
    for r in ranks:
        if r in ranks_mle or r in ranks_twonn or r in ranks_l2n2:
            if r not in used_ranks:
                used_ranks.append(r)

    # Re-index each series onto the unified x positions
    def reindex(series_vals, series_ranks):
        mapping = dict(zip(series_ranks, series_vals))
        return [mapping[r] for r in used_ranks]

    mle_plot = reindex(mle_f, ranks_mle)
    twonn_plot = reindex(twonn_f, ranks_twonn)
    l2n2_plot = reindex(l2n2_f, ranks_l2n2)

    x = np.arange(len(used_ranks))
    width = 0.25

    ax.bar(
        x - width,
        mle_plot,
        width,
        label="MLE"
    )

    ax.bar(
        x,
        twonn_plot,
        width,
        label="TWO-NN"
    )

    ax.bar(
        x + width,
        l2n2_plot,
        width,
        label="L2N2"
    )

    # Dataset references
    ax.axhline(
        dataset_refs["CIFAR10"]["MLE"],
        linestyle="--",
        linewidth=1,
        label="CIFAR10 MLE"
    )

    ax.axhline(
        dataset_refs["ImageNet100 Train"]["MLE"],
        linestyle=":",
        linewidth=1.5,
        label="ImageNet100 Train MLE"
    )

    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("Estimated Intrinsic Dimension")
    ax.set_title(
        f"Intrinsic Dimension vs Rank (Projection Dim={proj_dim})"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(used_ranks)

    ax.legend(
        frameon=False,
        ncol=2
    )

    plt.tight_layout()

    plt.savefig(
        f"intrinsic_dim_proj_{proj_dim}.pdf"
    )

    plt.savefig(
        f"intrinsic_dim_proj_{proj_dim}.png"
    )

    plt.show()

# --------------------------------------------------
# Generate plots
# --------------------------------------------------
plot_intrinsic_dimensions(
    mle_1024,
    twonn_1024,
    l2n2_1024,
    1024
)

plot_intrinsic_dimensions(
    mle_8192,
    twonn_8192,
    l2n2_8192,
    8192
)