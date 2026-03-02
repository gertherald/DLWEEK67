"""
train.py — Training loop for the DL Meta-Scorer
================================================
Trains WideDeepScorer on labelled review decision data.

Bootstrap target (available immediately):
  decision   — human reviewer choice: APPROVE / FLAG_FOR_REVIEW / BLOCK
  risk_score — numeric 0–100 composite score derived from signal severity

Long-term ground-truth targets (added retroactively once incidents are known):
  caused_production_incident, incident_severity, incident_type,
  days_until_incident, incident_linked_cwe
  (These are stored separately in INCIDENT_FEATURE_SCHEMA and not used here
   until the incident-linking pipeline populates them.)

Usage:
  python train.py --demo                      # synthetic data, quick start
  python train.py --data path/to/data.csv     # real labelled data
  python train.py --data data.csv --epochs 50 --batch 512
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_absolute_error

from feature_builder import FEATURE_NAMES, DECISION_MAP, DECISION_LABELS
from model import WideDeepScorer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_OUT  = os.path.join(SCRIPT_DIR, "dl_scorer.pt")
STATS_OUT  = os.path.join(SCRIPT_DIR, "norm_stats.json")


# ── Synthetic data generator ──────────────────────────────────────────────────

def generate_synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic training dataset to bootstrap training.
    Decision labels follow deterministic rules that reflect the feature plan,
    so the model can actually learn meaningful patterns.

    Replace or augment with real human-reviewer decisions as they accumulate.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for _ in range(n):
        # Tier 1 — ML outputs
        defect_prob   = float(rng.beta(2, 5))
        cvss_sev      = int(rng.choice([0, 1, 2, 3], p=[0.30, 0.35, 0.25, 0.10]))
        conf_vals     = rng.dirichlet([1, 1, 1, 1]).tolist()
        conf_variance = float(np.std(conf_vals))
        cross_agree   = int(cvss_sev >= 2 and defect_prob >= 0.5)
        num_cwes      = int(min(rng.poisson(0.8), 9))

        cwe_flags = {
            f"cwe_has_{name}": int(rng.random() < 0.15)
            for name in [
                "sql_injection", "xss", "buffer_overflow", "auth_bypass",
                "path_traversal", "improper_input", "use_after_free",
                "null_deref", "info_exposure",
            ]
        }

        attack_vec  = int(rng.choice([1, 2, 3], p=[0.30, 0.20, 0.50]))
        attack_cplx = int(rng.choice([0, 1],    p=[0.40, 0.60]))
        past_def    = int(rng.poisson(2))
        sa_warns    = int(rng.poisson(5))
        cyclo       = int(rng.integers(1, 30))
        test_cov    = float(rng.beta(5, 3))

        # Tier 2 — Human-in-the-loop
        user_overrode   = int(rng.random() < 0.30)
        override_dir    = int(rng.choice([-1, 0, 1], p=[0.40, 0.10, 0.50])) if user_overrode else 0
        override_acc    = float(rng.beta(6, 3))
        response_time   = float(rng.exponential(45))
        shadow_passed   = int(rng.choice([-1, 0, 1], p=[0.30, 0.20, 0.50]))
        feedback_sent   = float(rng.uniform(-0.3, 0.3))

        # Tier 3 — Code change context
        file_type    = int(rng.choice([0, 1, 2, 3, 4]))
        diff_added   = int(rng.exponential(50))
        diff_deleted = int(rng.exponential(20))
        is_new       = int(rng.random() < 0.15)
        auth_touch   = int(rng.random() < 0.20)
        db_touch     = int(rng.random() < 0.25)
        api_touch    = int(rng.random() < 0.30)
        imports      = int(rng.poisson(1))

        # Tier 4 — Enterprise context
        env          = int(rng.choice([0, 1, 2], p=[0.50, 0.30, 0.20]))
        svc_tier     = int(rng.choice([1, 2, 3], p=[0.20, 0.40, 0.40]))
        data_sens    = int(rng.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.15, 0.10]))
        compliance   = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7],
                                       p=[0.60, 0.10, 0.10, 0.05, 0.05, 0.03, 0.04, 0.03]))
        branch       = int(rng.choice([0, 1, 2], p=[0.65, 0.20, 0.15]))
        deadline     = int(rng.integers(1, 90))
        mod_defect   = float(rng.beta(2, 10))
        dev_defect   = float(rng.beta(2, 10))

        # Tier 5 — Instruction quality
        mentions_sec  = int(rng.random() < 0.25)
        align_score   = float(rng.beta(8, 2))
        call_count    = int(rng.poisson(3) + 1)
        blocked_count = int(rng.poisson(0.5))

        # ── Labelling rules ───────────────────────────────────────────────────
        # Risk accumulates from multiple independent signals.
        # These rules encode domain knowledge that the DL model must learn.
        risk = 0.0
        risk += defect_prob * 30                                   # high defect probability
        risk += cvss_sev * 10                                      # CVSS severity level
        risk += num_cwes * 3                                       # more CWEs = more risk
        risk += cwe_flags.get("cwe_has_sql_injection", 0) * 8
        risk += cwe_flags.get("cwe_has_auth_bypass", 0) * 10      # most critical CWE
        risk += cwe_flags.get("cwe_has_xss", 0) * 5
        risk += (1 - test_cov) * 8                                 # low coverage = risk
        risk += min(sa_warns, 20) * 0.5                            # static analysis
        if env == 2:                    risk += 15   # production target
        if svc_tier == 1:               risk += 10   # tier-1 service (payments/auth)
        if data_sens >= 2:              risk += 10   # PII or financial data
        if auth_touch:                  risk += 8    # auth module touched
        if db_touch and cwe_flags.get("cwe_has_sql_injection", 0):
                                        risk += 10   # DB touch + SQL = compound risk
        if shadow_passed == 0:          risk += 12   # simulation failed
        if shadow_passed == 1:          risk -= 10   # simulation passed = safer
        if user_overrode and override_dir == -1 and override_acc < 0.4:
                                        risk += 5    # inaccurate downgrade
        if response_time < 5:           risk += 5    # reviewed too quickly
        if compliance > 0 and cwe_flags.get("cwe_has_sql_injection", 0):
                                        risk += 10   # compliance + SQL
        if blocked_count >= 3:          risk += 8    # repeated blocks = pattern

        risk = float(np.clip(risk + rng.normal(0, 3), 0, 100))

        if risk < 30:
            decision = DECISION_MAP["APPROVE"]
        elif risk < 60:
            decision = DECISION_MAP["FLAG_FOR_REVIEW"]
        else:
            decision = DECISION_MAP["BLOCK"]

        rows.append({
            # Tier 1
            "defect_probability":               defect_prob,
            "cvss_severity_encoded":            cvss_sev,
            "cvss_confidence_critical":         conf_vals[0],
            "cvss_confidence_high":             conf_vals[1],
            "cvss_confidence_medium":           conf_vals[2],
            "cvss_confidence_low":              conf_vals[3],
            "cvss_confidence_variance":         conf_variance,
            "cross_model_agreement":            cross_agree,
            "num_cwes":                         num_cwes,
            **cwe_flags,
            "attack_vector_encoded":            attack_vec,
            "attack_complexity_encoded":        attack_cplx,
            "past_defects":                     past_def,
            "static_analysis_warnings":         sa_warns,
            "cyclomatic_complexity":            cyclo,
            "test_coverage":                    test_cov,
            # Tier 2
            "user_overrode_cvss":               user_overrode,
            "user_cvss_override_direction":     override_dir,
            "user_override_accuracy":           override_acc,
            "user_response_time_seconds":       response_time,
            "shadow_twin_passed":               shadow_passed,
            "user_feedback_sentiment":          feedback_sent,
            # Tier 3
            "file_type_encoded":                file_type,
            "diff_lines_added":                 diff_added,
            "diff_lines_deleted":               diff_deleted,
            "is_new_file":                      is_new,
            "touches_auth_module":              auth_touch,
            "touches_db_layer":                 db_touch,
            "touches_api_boundary":             api_touch,
            "new_imports_count":                imports,
            # Tier 4
            "deployment_environment":           env,
            "service_criticality_tier":         svc_tier,
            "data_sensitivity_level":           data_sens,
            "compliance_flags":                 compliance,
            "branch_type":                      branch,
            "days_to_release_deadline":         deadline,
            "module_defect_rate_30d":           mod_defect,
            "developer_recent_defect_rate":     dev_defect,
            # Tier 5
            "instruction_mentions_security":    mentions_sec,
            "instruction_code_alignment_score": align_score,
            "session_codex_call_count":         call_count,
            "consecutive_blocked_commits":      blocked_count,
            # Labels
            "risk_score": risk,
            "decision":   decision,
        })

    return pd.DataFrame(rows)


# ── Feature normalisation ─────────────────────────────────────────────────────

CONTINUOUS_FEATURES = [
    "defect_probability", "cvss_confidence_critical", "cvss_confidence_high",
    "cvss_confidence_medium", "cvss_confidence_low", "cvss_confidence_variance",
    "user_override_accuracy", "user_response_time_seconds", "user_feedback_sentiment",
    "test_coverage", "past_defects", "static_analysis_warnings", "cyclomatic_complexity",
    "diff_lines_added", "diff_lines_deleted", "new_imports_count",
    "days_to_release_deadline", "module_defect_rate_30d", "developer_recent_defect_rate",
    "instruction_code_alignment_score", "session_codex_call_count",
    "consecutive_blocked_commits",
]


def fit_normalize(df: pd.DataFrame) -> dict:
    """Compute min/max statistics for continuous features from training data."""
    stats = {}
    for col in CONTINUOUS_FEATURES:
        stats[col] = {"min": float(df[col].min()), "max": float(df[col].max())}
    return stats


def apply_normalize(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply min-max normalisation using pre-computed statistics."""
    df = df.copy()
    for col, s in stats.items():
        mn, mx = s["min"], s["max"]
        df[col] = (df[col] - mn) / (mx - mn + 1e-8)
    return df


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    df:         pd.DataFrame,
    epochs:     int   = 30,
    batch_size: int   = 256,
    lr:         float = 1e-3,
) -> tuple[WideDeepScorer, dict]:
    """Train the model. Returns (best_model, norm_stats)."""

    # 60 / 20 / 20 split
    X        = df[FEATURE_NAMES].values.astype(np.float32)
    y_dec    = df["decision"].values.astype(np.int64)
    y_risk   = df["risk_score"].values.astype(np.float32)

    X_tv, X_test, yd_tv, yd_test, yr_tv, yr_test = train_test_split(
        X, y_dec, y_risk, test_size=0.20, random_state=42, stratify=y_dec
    )
    X_train, X_val, yd_train, yd_val, yr_train, yr_val = train_test_split(
        X_tv, yd_tv, yr_tv, test_size=0.25, random_state=42, stratify=yd_tv
    )

    n = len(X)
    print(f"  Split  →  Train: {len(X_train)} ({len(X_train)/n*100:.0f}%)"
          f"  | Val: {len(X_val)} ({len(X_val)/n*100:.0f}%)"
          f"  | Test: {len(X_test)} ({len(X_test)/n*100:.0f}%)")

    def to_loader(Xa, yda, yra, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.tensor(Xa), torch.tensor(yda), torch.tensor(yra))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, yd_train, yr_train, shuffle=True)
    val_loader   = to_loader(X_val,   yd_val,   yr_val,   shuffle=False)

    model     = WideDeepScorer()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    ce_loss   = nn.CrossEntropyLoss()
    mse_loss  = nn.MSELoss()

    best_val_loss = float("inf")
    best_state    = None

    # Column headers
    print(f"\n{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>10}  {'Val Acc':>9}")
    print("-" * 48)

    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for Xb, ydb, yrb in train_loader:
            optimizer.zero_grad()
            risk_pred, dec_logits = model(Xb)
            # Combined loss: classification (primary) + risk MSE (0.01 weight)
            loss = ce_loss(dec_logits, ydb) + 0.01 * mse_loss(risk_pred, yrb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(Xb)

        # Validation
        model.eval()
        val_loss = 0.0
        correct  = 0
        with torch.no_grad():
            for Xb, ydb, yrb in val_loader:
                rp, dl    = model(Xb)
                val_loss += (ce_loss(dl, ydb) + 0.01 * mse_loss(rp, yrb)).item() * len(Xb)
                correct  += (dl.argmax(1) == ydb).sum().item()

        tl = train_loss / len(X_train)
        vl = val_loss   / len(X_val)
        va = correct    / len(X_val) * 100
        scheduler.step(vl)

        print(f"{epoch:>6}  {tl:>12.4f}  {vl:>10.4f}  {va:>8.2f}%")

        if vl < best_val_loss:
            best_val_loss = vl
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

    # Restore best checkpoint
    model.load_state_dict(best_state)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  TEST SET EVALUATION")
    print("=" * 55)

    model.eval()
    with torch.no_grad():
        risk_hat, dec_logits = model(torch.tensor(X_test))

    dec_pred     = dec_logits.argmax(1).numpy()
    risk_pred_np = risk_hat.numpy()

    decision_names = [DECISION_LABELS[i] for i in range(3)]
    print("\nDecision classification (3-class):")
    print(classification_report(yd_test, dec_pred, target_names=decision_names, zero_division=0))

    mae = mean_absolute_error(yr_test, risk_pred_np)
    print(f"Risk score MAE  : {mae:.2f} / 100")
    print("=" * 55)

    # Compute normalisation stats from training split for inference reuse
    norm_stats = fit_normalize(df.iloc[:len(X_train)])

    return model, norm_stats


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DL Meta-Scorer")
    parser.add_argument("--data",   type=str,   default=None, help="Path to CSV training data")
    parser.add_argument("--demo",   action="store_true",      help="Train on 5,000-row synthetic data")
    parser.add_argument("--epochs", type=int,   default=30)
    parser.add_argument("--batch",  type=int,   default=256)
    parser.add_argument("--lr",     type=float, default=1e-3)
    args = parser.parse_args()

    print("=" * 55)
    print("  DL Meta-Scorer — Training")
    print("=" * 55)

    if args.demo or args.data is None:
        print("\n[Data] Generating 5,000-row synthetic training dataset …")
        df = generate_synthetic_data(n=5000)
    else:
        print(f"\n[Data] Loading from '{args.data}' …")
        df = pd.read_csv(args.data)

    # Fit normalisation on all data, then apply
    stats = fit_normalize(df)
    df    = apply_normalize(df, stats)

    print(f"\n  Features : {len(FEATURE_NAMES)}")
    print(f"  Samples  : {len(df)}")
    print(f"  Decision distribution:")
    for label, count in df["decision"].value_counts().items():
        print(f"    {DECISION_LABELS[label]:<20} {count}")

    model, norm_stats = train(df, epochs=args.epochs, batch_size=args.batch, lr=args.lr)

    torch.save({"model_state": model.state_dict()}, MODEL_OUT)
    with open(STATS_OUT, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\n✓ Model saved      → {MODEL_OUT}")
    print(f"✓ Norm stats saved → {STATS_OUT}")
    print("\nRun:  python scorer.py --demo   to score a commit.")
