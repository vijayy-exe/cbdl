"""
Stage 9 — Clinical Evaluation Suite for the CBDL Pipeline.

Frozen evaluation using pretrained biomarkers (no clinical labels during training):
  - Ridge regression: biomarkers → UPDRS (RMSE, MAE, R²)
  - Logistic regression: biomarkers → Fall Risk binary (AUC, accuracy)
  - Spearman correlation: CBDi vs Motor Severity
  - Scatter plot: predicted vs true UPDRS
  - ROC curve: Fall Risk classification
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr


FEATURE_COLS = [
    "cbdi",
    "finger_gait_coupling",
    "neuromotor_stability",
    "coordination_index",
    "motor_variability",
    "synchronization_score",
]


def evaluate_updrs_regression(
    biomarkers_df: pd.DataFrame,
    output_dir: str = "output",
) -> Dict[str, float]:
    """
    Train a Ridge regressor on biomarkers → UPDRS using cross-validation.

    Args:
        biomarkers_df: DataFrame with biomarker columns and 'updrs' column.
        output_dir: Directory for output files.

    Returns:
        Dict with RMSE, MAE, R² metrics.
    """
    X = biomarkers_df[FEATURE_COLS].values
    y = biomarkers_df["updrs"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X_scaled, y, cv=cv)

    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    mae = float(mean_absolute_error(y, y_pred))
    r2 = float(r2_score(y, y_pred))

    metrics = {"rmse": rmse, "mae": mae, "r2": r2}

    # Save metrics
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/updrs_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Stage 9] UPDRS metrics: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.3f}")

    # Scatter plot
    _plot_updrs_scatter(y, y_pred, rmse, r2, output_dir)

    return metrics


def _plot_updrs_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rmse: float,
    r2: float,
    output_dir: str,
) -> None:
    """
    Save a scatter plot of predicted vs true UPDRS scores.

    Args:
        y_true: Ground truth UPDRS values.
        y_pred: Predicted UPDRS values.
        rmse: RMSE for display.
        r2: R² for display.
        output_dir: Output directory.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.5, color="#4C72B0", edgecolors="white", s=40)

    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("True UPDRS Score", fontsize=12)
    ax.set_ylabel("Predicted UPDRS Score", fontsize=12)
    ax.set_title(f"UPDRS Prediction: RMSE={rmse:.2f}, R²={r2:.3f}", fontsize=13, fontweight="bold")
    ax.legend()
    ax.spines[["right", "top"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/updrs_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Stage 9] UPDRS scatter plot saved → {output_dir}/updrs_scatter.png")


def evaluate_fall_risk(
    biomarkers_df: pd.DataFrame,
    output_dir: str = "output",
) -> Dict[str, float]:
    """
    Train a Logistic Regression on biomarkers → binary Fall Risk.

    Fall Risk is binarized: 0 if fall_risk ≤ 5, else 1.

    Args:
        biomarkers_df: DataFrame with biomarker and 'fall_risk' columns.
        output_dir: Directory for output files.

    Returns:
        Dict with AUC and accuracy metrics.
    """
    X = biomarkers_df[FEATURE_COLS].values
    y_raw = biomarkers_df["fall_risk"].values
    y = (y_raw > 5.0).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    if n_pos < 2 or n_neg < 2:
        # Degenerate case — use simple split
        split = len(y) // 2
        model.fit(X_scaled[:split], y[:split])
        y_prob = model.predict_proba(X_scaled[split:])[:, 1]
        y_pred = model.predict(X_scaled[split:])
        y_eval = y[split:]
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_prob = cross_val_predict(model, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        y_eval = y

    try:
        auc = float(roc_auc_score(y_eval, y_prob))
    except Exception:
        auc = 0.5
    acc = float(accuracy_score(y_eval, y_pred))

    metrics = {"auc": auc, "accuracy": acc}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{output_dir}/fallrisk_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Stage 9] Fall Risk metrics: AUC={auc:.3f}, Accuracy={acc:.3f}")

    # ROC curve
    _plot_roc_curve(y_eval, y_prob, auc, output_dir)

    return metrics


def _plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc: float,
    output_dir: str,
) -> None:
    """
    Save a ROC curve plot for Fall Risk classification.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities for the positive class.
        auc: AUC score for display.
        output_dir: Output directory.
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#E84855", linewidth=2.5, label=f"ROC Curve (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#E84855")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Fall Risk ROC Curve", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.spines[["right", "top"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fallrisk_roc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Stage 9] Fall Risk ROC plot saved → {output_dir}/fallrisk_roc.png")


def evaluate_cbdi_correlation(
    biomarkers_df: pd.DataFrame,
    output_dir: str = "output",
) -> pd.DataFrame:
    """
    Compute Spearman correlation between CBDi and Motor Severity.

    Args:
        biomarkers_df: DataFrame with 'cbdi' and 'motor_severity' columns.
        output_dir: Output directory.

    Returns:
        DataFrame with correlation results.
    """
    cbdi = biomarkers_df["cbdi"].values
    motor_sev = biomarkers_df["motor_severity"].values

    corr, pvalue = spearmanr(cbdi, motor_sev)

    result_df = pd.DataFrame([{
        "metric": "Spearman r (CBDi vs Motor Severity)",
        "correlation": float(corr),
        "p_value": float(pvalue),
        "n": len(biomarkers_df),
    }])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    result_df.to_csv(f"{output_dir}/correlation.csv", index=False)
    print(f"[Stage 9] CBDi–Motor Severity Spearman r={corr:.3f}, p={pvalue:.4f}")
    return result_df


def run_clinical_evaluation(
    biomarkers_df: pd.DataFrame,
    output_dir: str = "output",
) -> Dict:
    """
    Run the full clinical evaluation suite (Stage 9).

    Args:
        biomarkers_df: DataFrame with all biomarkers and clinical scores.
        output_dir: Directory to save all output files.

    Returns:
        Dict with all metric values.
    """
    print("\n[Stage 9] Running clinical evaluation…")
    updrs_metrics = evaluate_updrs_regression(biomarkers_df, output_dir)
    fallrisk_metrics = evaluate_fall_risk(biomarkers_df, output_dir)
    corr_df = evaluate_cbdi_correlation(biomarkers_df, output_dir)

    return {
        "updrs": updrs_metrics,
        "fall_risk": fallrisk_metrics,
        "cbdi_correlation": corr_df.to_dict(orient="records"),
    }
