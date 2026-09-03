import matplotlib.pyplot as plt
import pandas as pd

# =========================================================
# Data loading
# =========================================================
def load_curve(csv_path):
    """
    Expected CSV columns:
        - temperature
        - fpr_mean
        - fpr_std
    """

    df = pd.read_csv(csv_path)
    df = df.sort_values("temperature")

    # X-axis scaling (IMPORTANT FIX)
    x = df["temperature"].values * 0.5

    mean = df["fpr_mean"].values * 100
    std = df["fpr_std"].values * 100

    return x, mean, std


# =========================================================
# Style (ECCV / NeurIPS)
# =========================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "axes.linewidth": 1.1,
})

COLORS = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
]


def beautify(ax):
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# =========================================================
# EXPERIMENTS
# =========================================================
CNN_EXPERIMENTS = {
    "title": "(a) CNN",
    "paths": [
        "results/temp/CNN/densenet121_tinyimagenet/cmd/aggregated_results.csv",
        "results/temp/CNN/resnet34_tinyimagenet/cmd/aggregated_results.csv",
        "results/temp/CNN/wideresnet_tinyimagenet/cmd/aggregated_results.csv",
    ],
    "labels": ["DenseNet-121", "ResNet-34", "WideResNet-28x10"],
}

VIT_EXPERIMENTS = {
    "title": "(b) ViT",
    "paths": [
        "results/temp/ViT/fastvit_tinyimagenet/cmd/aggregated_results.csv",
        "results/temp/ViT/swin_tinyimagenet/cmd/aggregated_results.csv",
        "results/temp/ViT/vit_tinyimagenet/cmd/aggregated_results.csv",
    ],
    "labels": ["FastViT-T8", "Swin-Tiny", "ViT-Tiny"],
}

EXPERIMENTS = [CNN_EXPERIMENTS, VIT_EXPERIMENTS]


# =========================================================
# Plot
# =========================================================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)

for ax, exp in zip(axes, EXPERIMENTS):

    for i, (csv_path, label) in enumerate(zip(exp["paths"], exp["labels"])):

        x, mean, std = load_curve(csv_path)

        color = COLORS[i]

        ax.plot(
            x,
            mean,
            color=color,
            linewidth=2,
            marker="x",
            markersize=5,
            markeredgewidth=1.2,
            label=label,
        )

        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=color,
            alpha=0.18,
        )

    ax.set_title(exp["title"])
    ax.set_xlabel("Temperature (T)")
    beautify(ax)

    ax.legend(frameon=False, loc="upper right")

axes[0].set_ylim(25, 50)
# shared y-axis
axes[0].set_ylabel("FPR (%)")

plt.tight_layout()

# =========================================================
# Save
# =========================================================
plt.savefig("cnn_vit_temperature_fpr.pdf", bbox_inches="tight")
plt.savefig("cnn_vit_temperature_fpr.png", dpi=400, bbox_inches="tight")

plt.show()