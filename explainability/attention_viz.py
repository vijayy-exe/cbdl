"""
Stage 8b — Attention Map Visualization for the CBDL Pipeline.

Plots the cross-limb attention weights as a 4×4 heatmap and saves
to /output/attention_map.png.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


STREAM_NAMES = ["Finger", "Gait", "Balance", "Wrist/Sensor"]


def plot_attention_map(
    attn_weights: np.ndarray,
    output_path: str = "output/attention_map.png",
    title: str = "Cross-Limb Attention Map",
) -> None:
    """
    Plot the 4×4 cross-limb attention weight matrix as a heatmap.

    Args:
        attn_weights: Attention weight array of shape (4, 4) or (N, 4, 4).
                      If 3-D, the mean across the first axis is used.
        output_path: Path to save the output PNG.
        title: Plot title.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if attn_weights.ndim == 3:
        matrix = attn_weights.mean(axis=0)
    else:
        matrix = attn_weights.copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Main heatmap
    sns.heatmap(
        matrix,
        ax=axes[0],
        xticklabels=STREAM_NAMES,
        yticklabels=STREAM_NAMES,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={"label": "Attention Weight"},
        linewidths=0.5,
        square=True,
        vmin=0.0,
        vmax=matrix.max(),
    )
    axes[0].set_title("Mean Cross-Limb Attention Weights")
    axes[0].set_xlabel("Target Stream")
    axes[0].set_ylabel("Source Stream")

    # Off-diagonal bar chart (dependency strengths)
    n = matrix.shape[0]
    labels, values = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                labels.append(f"{STREAM_NAMES[i][:3]}→{STREAM_NAMES[j][:3]}")
                values.append(matrix[i, j])

    colors = plt.cm.YlOrRd(np.array(values) / (max(values) + 1e-8))
    axes[1].bar(range(len(labels)), values, color=colors, edgecolor="white")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Attention Weight")
    axes[1].set_title("Cross-Body Dependency Strengths")
    axes[1].spines[["right", "top"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Stage 8] Attention map saved → {output_path}")
