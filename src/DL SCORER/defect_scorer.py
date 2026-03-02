"""
defect_scorer.py — Defect ML inference for freshly generated code
=================================================================
Wraps the saved Logistic Regression model (lr_model_top5.pkl) so that
defect probability can be computed for any code snippet without needing
to re-run the full Defect ML pipeline.

The Defect ML model requires StandardScaler-normalised inputs.  Since the
scaler is not saved alongside the model, this module re-fits the same
scaler from the training split of the original dataset — which is fully
deterministic (same random_state=42, same split).

Usage:
    from defect_scorer import predict_defect

    result = predict_defect({
        "cyclomatic_complexity":    8,
        "static_analysis_warnings": 4,
        "past_defects":             0,
        "response_for_class":       12,
        "test_coverage":            0.0,
    })
    print(result["defect_probability"])   # e.g. 0.68
    print(result["risk_band"])            # e.g. "🟠 MEDIUM-HIGH"
"""

import os
import pickle
import functools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATASET_PATH    = os.path.join(PROJECT_ROOT, "data", "software_defect_prediction_dataset.csv")
MODEL_TOP5_PATH = os.path.join(SCRIPT_DIR, "..", "DEFECT ML", "lr_model_top5.pkl")
MODEL_FULL_PATH = os.path.join(SCRIPT_DIR, "..", "DEFECT ML", "lr_model_full.pkl")

TOP5_FEATURES = [
    "past_defects",
    "static_analysis_warnings",
    "cyclomatic_complexity",
    "response_for_class",
    "test_coverage",
]

ALL_FEATURES = [
    "lines_of_code", "cyclomatic_complexity", "num_functions", "num_classes",
    "comment_density", "code_churn", "developer_experience_years", "num_developers",
    "commit_frequency", "bug_fix_commits", "past_defects", "test_coverage",
    "duplication_percentage", "avg_function_length", "depth_of_inheritance",
    "response_for_class", "coupling_between_objects", "lack_of_cohesion",
    "build_failures", "static_analysis_warnings", "security_vulnerabilities",
    "performance_issues",
]

DECISION_THRESHOLD = 0.30   # matches Defect ML config

_RISK_BANDS = [
    (0.30, "🟢 LOW"),
    (0.55, "🟡 MEDIUM"),
    (0.75, "🟠 HIGH"),
    (1.01, "🔴 CRITICAL"),
]


# ── Lazy-loaded model + scaler (computed once, cached) ────────────────────────

@functools.lru_cache(maxsize=2)
def _load(use_top5: bool = True) -> tuple:
    """
    Load the LR model and re-fit the StandardScaler from the training split.
    Result is cached so the dataset is only read once per process.
    """
    model_path     = MODEL_TOP5_PATH if use_top5 else MODEL_FULL_PATH
    feature_cols   = TOP5_FEATURES   if use_top5 else ALL_FEATURES

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Re-fit scaler on the same training split used during model training
    df      = pd.read_csv(DATASET_PATH)
    X       = df[feature_cols]
    y       = df["defect"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    scaler = StandardScaler().fit(X_train)

    return model, scaler, feature_cols


# ── Public API ────────────────────────────────────────────────────────────────

def predict_defect(
    feature_values: dict,
    use_top5: bool = True,
    threshold: float = DECISION_THRESHOLD,
) -> dict:
    """
    Predict defect probability for a single code instance.

    Parameters
    ----------
    feature_values : dict
        Keys from TOP5_FEATURES (or ALL_FEATURES if use_top5=False).
        Missing keys default to 0.
    use_top5 : bool
        True  → use lr_model_top5.pkl  (5 features, default)
        False → use lr_model_full.pkl  (22 features)
    threshold : float
        Decision threshold (default 0.30 matches Defect ML config).

    Returns
    -------
    dict with:
      defect_probability — float 0–1
      prediction         — 0 (no defect) or 1 (defect)
      risk_band          — human-readable risk level
      feature_values     — dict of features actually used
    """
    model, scaler, cols = _load(use_top5)

    x        = pd.DataFrame([[feature_values.get(c, 0) for c in cols]], columns=cols)
    x_scaled = scaler.transform(x)

    prob_defect = float(model.predict_proba(x_scaled)[0][1])
    prediction  = int(prob_defect > threshold)

    risk_band = next(
        label for bound, label in _RISK_BANDS if prob_defect < bound
    )

    return {
        "defect_probability": round(prob_defect, 4),
        "prediction":         prediction,
        "risk_band":          risk_band,
        "feature_values":     {c: feature_values.get(c, 0) for c in cols},
    }


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    # High-risk profile: lots of past defects, warnings, complex, untested
    high_risk = {
        "past_defects":             8,
        "static_analysis_warnings": 15,
        "cyclomatic_complexity":    28,
        "response_for_class":       35,
        "test_coverage":            0.0,
    }
    # Low-risk profile: clean, well-tested code
    low_risk = {
        "past_defects":             0,
        "static_analysis_warnings": 0,
        "cyclomatic_complexity":    4,
        "response_for_class":       8,
        "test_coverage":            0.85,
    }
    for label, fv in [("HIGH-RISK", high_risk), ("LOW-RISK", low_risk)]:
        r = predict_defect(fv)
        print(f"{label:12} → P(defect)={r['defect_probability']:.3f}  {r['risk_band']}")
