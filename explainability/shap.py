"""
Stage 8a — SHAP Explainability for the CBDL Pipeline.

Uses KernelSHAP over the 6 biomarkers to explain UPDRS predictions.
Outputs a SHAP summary plot to /output/shap_summary.png.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from typing import Optional
from sklearn.linear_model import Ridge


def compute_shap_values(
    biomarkers_df: pd.DataFrame,
    target_col: str = "updrs",
    feature_cols: Optional[list] = None,
    n_background: int = 50,
) -> tuple:
    """
    Train a Ridge regressor on biomarkers and compute KernelSHAP values.

    Args:
        biomarkers_df: DataFrame with biomarker columns and target column.
        target_col: Name of the target column (e.g. 'updrs').
        feature_cols: List of feature column names. If None, uses default 6.
        n_background: Number of background samples for KernelSHAP.

    Returns:
        Tuple of (shap_values, explainer, X_features) for further use.
    """
    if feature_cols is None:
        feature_cols = [
            "cbdi",
            "finger_gait_coupling",
            "neuromotor_stability",
            "coordination_index",
            "motor_variability",
            "synchronization_score",
        ]

    X = biomarkers_df[feature_cols].values.astype(np.float64)
    y = biomarkers_df[target_col].values.astype(np.float64)

    # Train ridge regressor
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    pred_fn = lambda x: model.predict(x)

    # KernelSHAP — use subset as background
    n_bg = min(n_background, len(X))
    background = X[:n_bg]
    explainer = shap.KernelExplainer(pred_fn, background)

    # Compute SHAP values for all samples (use small nsamples for speed)
    shap_values = explainer.shap_values(X, nsamples=100, silent=True)

    return shap_values, explainer, X, feature_cols, model


def plot_shap_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    output_path: str = "output/shap_summary.png",
) -> None:
    """
    Create and save a SHAP beeswarm/bar summary plot.

    Args:
        shap_values: SHAP values array of shape (N, n_features).
        X: Feature matrix of shape (N, n_features).
        feature_names: List of feature name strings.
        output_path: Path to save the PNG.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("SHAP Feature Importance — UPDRS Prediction", fontsize=14, fontweight="bold")

    # Bar plot (mean |SHAP|)
    mean_abs = np.abs(shap_values).mean(axis=0)
    sorted_idx = np.argsort(mean_abs)[::-1]
    axes[0].barh(
        [feature_names[i] for i in sorted_idx[::-1]],
        mean_abs[sorted_idx[::-1]],
        color="#4C72B0",
        edgecolor="white",
    )
    axes[0].set_xlabel("Mean |SHAP Value|")
    axes[0].set_title("Global Feature Importance")
    axes[0].spines[["right", "top"]].set_visible(False)

    # Scatter: SHAP value vs feature value for each biomarker
    colors = plt.cm.RdBu(np.linspace(0, 1, len(feature_names)))
    for i, (fname, color) in enumerate(zip(feature_names, colors)):
        axes[1].scatter(
            X[:, i],
            shap_values[:, i],
            alpha=0.4,
            label=fname,
            s=10,
            color=color,
        )
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("Feature Value")
    axes[1].set_ylabel("SHAP Value (impact on UPDRS)")
    axes[1].set_title("SHAP Values vs Feature Values")
    axes[1].legend(loc="upper right", fontsize=7)
    axes[1].spines[["right", "top"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Stage 8] SHAP summary plot saved → {output_path}")


def run_shap_analysis(
    biomarkers_df: pd.DataFrame,
    output_path: str = "output/shap_summary.png",
    n_background: int = 50,
) -> None:
    """
    Full SHAP pipeline: fit model, compute SHAP values, and save plot.

    Args:
        biomarkers_df: DataFrame with biomarkers and 'updrs' column.
        output_path: Output path for the SHAP summary PNG.
        n_background: KernelSHAP background sample count.
    """
    shap_vals, explainer, X, feature_cols, model = compute_shap_values(
        biomarkers_df,
        target_col="updrs",
        n_background=n_background,
    )
    plot_shap_summary(shap_vals, X, feature_cols, output_path)
