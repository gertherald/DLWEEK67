# ============================================================
# evaluate.py — Metrics, plots, and single-instance prediction
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, average_precision_score,
)

from config import DECISION_THRESHOLD, RISK_BANDS, PLOT_LR_RESULTS, PLOT_FEATURE_IMP


# ── Core metrics ─────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Return a dict of all key scalar metrics."""
    return {
        "accuracy"   : accuracy_score(y_true, y_pred),
        "roc_auc"    : roc_auc_score(y_true, y_proba),
        "precision_0": precision_score(y_true, y_pred, pos_label=0),
        "precision_1": precision_score(y_true, y_pred, pos_label=1),
        "recall_0"   : recall_score(y_true, y_pred, pos_label=0),
        "recall_1"   : recall_score(y_true, y_pred, pos_label=1),
        "f1_0"       : f1_score(y_true, y_pred, pos_label=0),
        "f1_1"       : f1_score(y_true, y_pred, pos_label=1),
        "macro_f1"   : f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "mAP"        : average_precision_score(y_true, y_proba),
    }


def print_metrics(metrics: dict, threshold: float = DECISION_THRESHOLD) -> None:
    """Pretty-print all scalar metrics."""
    w = 30
    print("=" * 60)
    print(f"   METRICS SUMMARY  (Threshold = {threshold})")
    print("=" * 60)
    print(f"{'Metric':<{w}} {'No Defect (0)':>14} {'Defect (1)':>12}")
    print("-" * 60)
    for key, label in [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 Score")]:
        v0 = metrics[f"{key}_0"] * 100
        v1 = metrics[f"{key}_1"] * 100
        print(f"{label:<{w}} {v0:>13.2f}% {v1:>11.2f}%")
    print("=" * 60)
    print(f"{'Accuracy':<{w}} {metrics['accuracy']*100:>13.2f}%")
    print(f"{'ROC-AUC':<{w}} {metrics['roc_auc']:>14.4f}")
    print(f"{'Macro F1':<{w}} {metrics['macro_f1']*100:>13.2f}%")
    print(f"{'Weighted F1':<{w}} {metrics['weighted_f1']*100:>13.2f}%")
    print(f"{'mAP':<{w}} {metrics['mAP']*100:>13.2f}%")
    print("=" * 60)


# ── Plots ────────────────────────────────────────────────────

def plot_confusion_and_roc(
    y_true, y_pred, y_proba,
    title_suffix: str = "Logistic Regression",
    save_path: str = PLOT_LR_RESULTS,
) -> None:
    """Plot confusion matrix + ROC curve side-by-side and save to disk."""
    roc_auc = roc_auc_score(y_true, y_proba)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["No Defect", "Defect"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues"
    )
    axes[0].set_title(f"Confusion Matrix — {title_suffix}", fontsize=14, fontweight="bold")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[1].plot(fpr, tpr, color="darkorange", lw=2,
                 label=f"ROC (AUC = {roc_auc:.4f})")
    axes[1].plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    axes[1].set_xlabel("False Positive Rate", fontsize=12)
    axes[1].set_ylabel("True Positive Rate", fontsize=12)
    axes[1].set_title(f"ROC Curve — {title_suffix}", fontsize=14, fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Plot saved as '{save_path}'")


def plot_feature_importance(
    model: LogisticRegression,
    feature_names: list[str],
    top_n: int = 15,
    save_path: str = PLOT_FEATURE_IMP,
) -> pd.DataFrame:
    """
    Bar chart of the top-N features by absolute LR coefficient.
    Returns the full coefficient DataFrame (sorted by importance).
    """
    coef_df = (
        pd.DataFrame({
            "Feature"        : feature_names,
            "Coefficient"    : model.coef_[0],
            "Abs_Coefficient": np.abs(model.coef_[0]),
        })
        .sort_values("Abs_Coefficient", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\nTop {top_n} features (by |coefficient|):")
    print(coef_df.head(top_n).to_string(index=False))

    top = coef_df.head(top_n).sort_values("Coefficient")
    colors = ["#d73027" if c > 0 else "#4575b4" for c in top["Coefficient"]]

    plt.figure(figsize=(10, 7))
    plt.barh(top["Feature"], top["Coefficient"],
             color=colors, edgecolor="white", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Logistic Regression Coefficient", fontsize=12)
    plt.title(
        f"Top {top_n} Feature Importances\n"
        "(Red = Increases Defect Risk, Blue = Decreases)",
        fontsize=13, fontweight="bold",
    )
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✓ Plot saved as '{save_path}'")
    return coef_df


# ── Single-instance prediction ────────────────────────────────

def predict_single_instance(
    model: LogisticRegression,
    X_test_scaled: np.ndarray,
    X_test: pd.DataFrame,
    y_test,
    row_index: int = 0,
) -> None:
    """Print a detailed prediction report for one test-set instance."""
    instance = X_test_scaled[row_index].reshape(1, -1)
    actual   = y_test.iloc[row_index]

    proba           = model.predict_proba(instance)[0]
    prob_no_defect  = proba[0]
    prob_defect     = proba[1]
    prediction      = model.predict(instance)[0]

    # Risk level
    risk_level = next(
        label for threshold, label in RISK_BANDS if prob_defect < threshold
    )

    print(f"\nFeature values for test-set row {row_index}:")
    print(X_test.iloc[row_index].to_string())
    print("\n" + "=" * 50)
    print("         PREDICTION vs ACTUAL")
    print("=" * 50)
    print(f"  P(No Defect) : {prob_no_defect*100:.2f}%")
    print(f"  P(Defect)    : {prob_defect*100:.2f}%")
    print(f"  Risk Level   : {risk_level}")
    print(f"  Prediction   : {'DEFECT' if prediction == 1 else 'NO DEFECT'}")
    print(f"  Actual       : {'DEFECT' if actual == 1 else 'NO DEFECT'}")
    print(f"  Correct?     : {'✅ YES' if prediction == actual else '❌ NO'}")
    print("=" * 50)


# ── Model comparison table ────────────────────────────────────

def build_comparison_table(
    metrics_full: dict,
    metrics_top5: dict,
    label_full: str = "22 Features",
    label_top5: str = "Top 5 Features",
) -> pd.DataFrame:
    """Pretty-print and return a comparison DataFrame for two model variants."""
    keys = [
        ("accuracy",    "Accuracy",             lambda v: f"{v*100:.2f}%"),
        ("roc_auc",     "ROC-AUC",              lambda v: f"{v:.4f}"),
        ("precision_1", "Precision (Defect)",   lambda v: f"{v*100:.2f}%"),
        ("recall_1",    "Recall (Defect)",      lambda v: f"{v*100:.2f}%"),
        ("f1_1",        "F1 Score (Defect)",    lambda v: f"{v*100:.2f}%"),
        ("mAP",         "mAP",                  lambda v: f"{v*100:.2f}%"),
    ]

    rows = []
    for key, label, fmt in keys:
        rows.append({
            "Metric"   : label,
            label_full : fmt(metrics_full[key]),
            label_top5 : fmt(metrics_top5[key]),
        })

    table = pd.DataFrame(rows).set_index("Metric")

    print("=" * 55)
    print("   MODEL COMPARISON (Threshold = 0.30, Test Set)")
    print("=" * 55)
    print(table.to_string())
    print("=" * 55)
    return table
