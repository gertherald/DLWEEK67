# ============================================================
# config.py — Central configuration for the defect prediction pipeline
# ============================================================

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Dataset ────────────────────────────────────────────────
DATASET_FILE     = "data/software_defect_prediction_dataset.csv"
TARGET_COLUMN    = "defect"
KAGGLE_DATASET   = "mirzayasirabdullah07/software-defect-prediction-dataset"

# ── Train / Test split ─────────────────────────────────────
TEST_SIZE        = 0.2
RANDOM_STATE     = 42

# ── Model ──────────────────────────────────────────────────
SOLVER           = "lbfgs"
MAX_ITER         = 1000
CLASS_WEIGHT     = "balanced"   # handles class imbalance

# ── Prediction threshold ───────────────────────────────────
DECISION_THRESHOLD = 0.30       # Lower than 0.5 to boost recall on defect class

# ── Feature subsets ────────────────────────────────────────
TOP5_FEATURES = [
    "past_defects",
    "static_analysis_warnings",
    "cyclomatic_complexity",
    "response_for_class",
    "test_coverage",
]

# ── Risk bands (based on defect probability) ───────────────
RISK_BANDS = [
    (0.40, "🟢 LOW RISK"),
    (0.70, "🟡 MEDIUM RISK"),
    (1.01, "🔴 HIGH RISK"),
]

# ── Output paths (all saved into the DEFECT ML folder) ─────
PLOT_LR_RESULTS      = os.path.join(SCRIPT_DIR, "logistic_regression_results.png")
PLOT_FEATURE_IMP     = os.path.join(SCRIPT_DIR, "feature_importance.png")
PLOT_TOP5_RESULTS    = os.path.join(SCRIPT_DIR, "top5_features_results.png")
MODEL_FULL_PATH      = os.path.join(SCRIPT_DIR, "lr_model_full.pkl")
MODEL_TOP5_PATH      = os.path.join(SCRIPT_DIR, "lr_model_top5.pkl")
