"""
train_model.py
--------------
Trains an XGBoost classifier to predict CVSS severity (LOW/MEDIUM/HIGH/CRITICAL).

Architecture:
  INPUTS  → CVSS contextual signals + all regex-detectable CWE flags from analyse_code()
  TARGET  → cvss_severity  (single multi-class label)

Split: 60% train / 20% validation / 20% test

Run:
  python train_model.py
"""

import os
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "../../data/ml_features_readable_WITH_all_word_features.csv")
MODEL_OUT  = os.path.join(SCRIPT_DIR, "severity_model.pkl")
INPUT_OUT  = os.path.join(SCRIPT_DIR, "input_cols.pkl")

# ── Model inputs ───────────────────────────────────────────────────────────────
# CVSS contextual signals + all CWE flags detectable by regex in analyse_code().
# Including the target-correlated CWE flags means SQL code → cwe_has_sql_injection=1,
# XSS code → cwe_has_xss=1, giving the model distinct input vectors per vuln type.
INPUT_COLS = [
    # CVSS contextual
    'attack_vector_encoded',
    'attack_complexity_encoded',
    'privileges_required_encoded',
    'user_interaction_encoded',
    'vuln_status_encoded',
    'has_configurations',
    'num_references',
    'num_cwes',
    # All regex-detectable CWE flags
    'cwe_has_sql_injection',
    'cwe_has_xss',
    'cwe_has_buffer_overflow',
    'cwe_has_path_traversal',
    'cwe_has_improper_input',
    'cwe_has_use_after_free',
    'cwe_has_null_deref',
    'cwe_has_auth_bypass',
    'cwe_has_info_exposure',
]

# ── Encoding maps  (must match analyse_code() in codex_risk_pipeline.py) ───────
ATTACK_VECTOR_MAP = {
    'NETWORK': 3, 'ADJACENT_NETWORK': 2, 'ADJACENT': 2, 'LOCAL': 1, 'PHYSICAL': 0
}
ATTACK_COMPLEXITY_MAP   = {'LOW': 1, 'HIGH': 0}
PRIVILEGES_REQUIRED_MAP = {'NONE': 2, 'LOW': 1, 'HIGH': 0}
USER_INTERACTION_MAP    = {'NONE': 1, 'REQUIRED': 0, 'ACTIVE': 0, 'PASSIVE': 0}
VULN_STATUS_MAP         = {
    'Analyzed': 1, 'Modified': 1,
    'Awaiting Analysis': 0, 'Undergoing Analysis': 0, 'Received': 0
}


def load_and_prepare(path: str):
    df = pd.read_csv(path, low_memory=False)
    df = df[df['cvss_severity'] != 'NONE'].copy()

    # Encode CVSS categoricals → _encoded names matching analyse_code() output
    df['attack_vector_encoded']       = df['attack_vector'].map(ATTACK_VECTOR_MAP).fillna(1)
    df['attack_complexity_encoded']   = df['attack_complexity'].map(ATTACK_COMPLEXITY_MAP).fillna(1)
    df['privileges_required_encoded'] = df['privileges_required'].map(PRIVILEGES_REQUIRED_MAP).fillna(2)
    df['user_interaction_encoded']    = df['user_interaction'].map(USER_INTERACTION_MAP).fillna(1)
    df['vuln_status_encoded']         = df['vuln_status'].map(VULN_STATUS_MAP).fillna(0)

    df = df.fillna(0)

    X     = df[INPUT_COLS].astype(float)
    y_raw = df['cvss_severity']

    return X, y_raw.reset_index(drop=True)


def _print_eval(model, le, X, y, label: str):
    y_pred_idx = model.predict(X)
    y_pred     = le.inverse_transform(y_pred_idx)
    y_true     = le.inverse_transform(y)
    print(f"\n  {label} results ({len(X)} rows):")
    print(classification_report(y_true, y_pred, target_names=le.classes_, zero_division=0))


def train(X: pd.DataFrame, y_raw: pd.Series):
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)

    # ── 60 / 20 / 20 split ────────────────────────────────────────────────────
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    n = len(X)
    print(f"  Split  →  Train: {len(X_train)} ({len(X_train)/n*100:.0f}%)  "
          f"| Val: {len(X_val)} ({len(X_val)/n*100:.0f}%)  "
          f"| Test: {len(X_test)} ({len(X_test)/n*100:.0f}%)")

    # ── Train ──────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Training XGBoost  "
          f"({len(INPUT_COLS)} inputs → cvss_severity)...")

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        eval_metric='mlogloss',
        random_state=42,
        n_jobs=1,
    )
    model.fit(X_train, y_train)

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n[2/3] Validation set (20%):")
    _print_eval(model, le, X_val, y_val, "Validation")

    # ── Test ──────────────────────────────────────────────────────────────────
    print("[3/3] Test set (20%) — final held-out evaluation:")
    _print_eval(model, le, X_test, y_test, "Test")

    return model, le


if __name__ == "__main__":
    print("=" * 60)
    print("  CVE Severity Model — XGBoost Training")
    print("=" * 60)

    print(f"\nLoading data from:\n  {DATA_PATH}\n")
    X, y_raw = load_and_prepare(DATA_PATH)

    print(f"Dataset : {X.shape[0]} rows  |  {len(INPUT_COLS)} inputs")
    print(f"Target  : cvss_severity")
    print(f"Classes : {dict(y_raw.value_counts())}\n")

    model, le = train(X, y_raw)

    bundle = {'model': model, 'classes': le.classes_}
    joblib.dump(bundle,     MODEL_OUT)
    joblib.dump(INPUT_COLS, INPUT_OUT)

    print(f"\nSaved model      → {MODEL_OUT}")
    print(f"Saved input cols → {INPUT_OUT}")
    print("\nDone. Run codex_risk_pipeline.py to use the model.")
