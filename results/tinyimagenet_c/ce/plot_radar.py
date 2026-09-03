import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# =========================================================
# CONFIG
# =========================================================

ROOT = "tinyimagenet_c/ce"

ARCHS = [
    "resnet34_tinyimagenet",
    "swin_tinyimagenet"
]

# PLACEHOLDER TITLES (EDIT THESE FOR PAPER)
SUBPLOT_TITLES = [
    "(a) ResNet-34",
    "(b) Swin-Tiny"
]

N_METHODS = 3
N_SEVERITIES = 5

METHOD_COLORS = ["#C44E52", "#55A868", "#4C72B0"]

FMIN, FMAX = 40.0, 60.0

FONT = 1.05


# =========================================================
# Read metrics
# =========================================================
def read_metrics(file_path):

    df = pd.read_csv(file_path)
    df = df.set_index("metric")["value"]

    return {
        "fpr": float(df["fpr_mean"]),
        "acc": float(df["accuracy_mean"]),
        "acc_std": float(df["accuracy_std"]),
    }


# =========================================================
# Load method
# =========================================================
def load_method(method_path):

    fpr_vals, acc_vals, acc_std_vals = [], [], []

    for s in range(1, N_SEVERITIES + 1):

        file_path = os.path.join(method_path, f"res{s}.csv")

        if os.path.exists(file_path):
            m = read_metrics(file_path)
        else:
            m = {"fpr": np.nan, "acc": np.nan, "acc_std": np.nan}

        fpr_vals.append(m["fpr"])
        acc_vals.append(m["acc"])
        acc_std_vals.append(m["acc_std"])

    return (
        np.array(fpr_vals, dtype=np.float32),
        np.array(acc_vals, dtype=np.float32),
        np.array(acc_std_vals, dtype=np.float32),
    )


# =========================================================
# Load architecture
# =========================================================
def load_arch(arch_path):

    methods = sorted([
        d for d in os.listdir(arch_path)
        if os.path.isdir(os.path.join(arch_path, d))
    ])[:N_METHODS]

    data = {}

    for m in methods:
        data[m] = load_method(os.path.join(arch_path, m))

    return data


# =========================================================
# Normalize FPR (fixed scale)
# =========================================================
def norm(x):
    x = np.clip(x, FMIN, FMAX)
    return (x - FMIN) / (FMAX - FMIN + 1e-8)


# =========================================================
# Radar
# =========================================================
def radar(ax, values, angles, color, label):

    values = np.concatenate([values, [values[0]]])
    ax.plot(angles, values, lw=2, color=color)
    ax.fill(angles, values, alpha=0.15, color=color)


# =========================================================
# PLOT
# =========================================================
def plot():

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 5),
        subplot_kw=dict(polar=True)
    )

    angles = np.linspace(0, 2*np.pi, N_SEVERITIES, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------
    data_all = {}

    for arch in ARCHS:
        data_all[arch] = load_arch(os.path.join(ROOT, arch))

    # =====================================================
    # PLOT EACH SUBFIGURE
    # =====================================================
    for ax, arch, title in zip(axes, ARCHS, SUBPLOT_TITLES):

        data = data_all[arch]

        acc_ref = None

        for i, (method, (fpr, acc, acc_std)) in enumerate(sorted(data.items())):

            radar(
                ax,
                norm(fpr),
                angles,
                METHOD_COLORS[i],
                method
            )

            if acc_ref is None:
                acc_ref = (acc, acc_std)

        # -------------------------------------------------
        # severity labels with accuracy
        # -------------------------------------------------
        acc, acc_std = acc_ref

        labels = [
            f"Severity {i+1}\n{acc[i]:.1f}±{acc_std[i]:.1f}"
            for i in range(N_SEVERITIES)
        ]

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9 * FONT)

        ax.set_title(title, fontsize=12, pad=18)

        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)

        # radial ticks (30–80%)
        levels = np.linspace(FMIN, FMAX, 6)
        radii = (levels - FMIN) / (FMAX - FMIN + 1e-8)

        ax.set_yticks(radii)
        ax.set_yticklabels([f"{v:.0f}%" for v in levels], fontsize=9)

# =========================================================
# SINGLE GLOBAL LEGEND
# =========================================================
    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS[2], lw=2, label="CMD"),
        Line2D([0], [0], color=METHOD_COLORS[0], lw=2, label="Doctor"),
        Line2D([0], [0], color=METHOD_COLORS[1], lw=2, label=r"LogitGap$_3$"),
    ]

    axes[1].legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.25, 1.1),
        frameon=False,
        ncol=1
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # =====================================================
    # SAVE
    # =====================================================
    plt.savefig("severity_radar_fpr.pdf", bbox_inches="tight")
    plt.savefig("severity_radar_fpr.png", dpi=400, bbox_inches="tight")

    plt.show()


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    plot()