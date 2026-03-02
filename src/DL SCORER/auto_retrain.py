"""
auto_retrain.py — Automatic retraining for the DL Meta-Scorer
==============================================================
Loads all accumulated human reviews from reviews.csv, combines them
with 5,000 synthetic rows (real rows weighted ×5 so they aren't drowned
out), and retrains the sklearn MLPClassifier.  Saves the updated model
and normalisation stats.

No PyTorch dependency — works standalone.

Public API:
    from auto_retrain import run_retrain
    result = run_retrain(verbose=True)
    # → {"val_accuracy": 0.80, "total_reviews": 40, "accuracy_delta": +0.05}

CLI:
    python auto_retrain.py              # retrain now on all accumulated reviews
    python auto_retrain.py --status     # show retrain_state.json history
    python auto_retrain.py --threshold 50  # retrain only if ≥50 pending reviews
"""

import os
import sys
import json
import argparse
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
REVIEWS_CSV   = os.path.join(SCRIPT_DIR, "reviews.csv")
STATE_JSON    = os.path.join(SCRIPT_DIR, "retrain_state.json")
MODEL_OUT     = os.path.join(SCRIPT_DIR, "dl_scorer_sklearn.pkl")
STATS_OUT     = os.path.join(SCRIPT_DIR, "norm_stats_sklearn.json")

sys.path.insert(0, SCRIPT_DIR)
from feature_builder import FEATURE_NAMES, DECISION_LABELS  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
N_SYNTHETIC        = 5000   # synthetic rows generated each retrain
REAL_WEIGHT        = 5      # real review rows repeated this many times
SHADOW_FAIL_WEIGHT = 3      # shadow-failed real rows weighted REAL_WEIGHT × this

# Continuous features that get min-max normalised
_CONTINUOUS = [
    "defect_probability", "cvss_confidence_critical", "cvss_confidence_high",
    "cvss_confidence_medium", "cvss_confidence_low", "cvss_confidence_variance",
    "user_override_accuracy", "user_response_time_seconds", "user_feedback_sentiment",
    "test_coverage", "past_defects", "static_analysis_warnings", "cyclomatic_complexity",
    "diff_lines_added", "diff_lines_deleted", "new_imports_count",
    "days_to_release_deadline", "module_defect_rate_30d", "developer_recent_defect_rate",
    "instruction_code_alignment_score", "session_codex_call_count",
    "consecutive_blocked_commits",
]


# ── Normalisation helpers ──────────────────────────────────────────────────────

def fit_normalize(df: pd.DataFrame) -> dict:
    """Compute min-max stats for continuous features from df."""
    return {
        c: {"min": float(df[c].min()), "max": float(df[c].max())}
        for c in _CONTINUOUS if c in df.columns
    }


def apply_normalize(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply saved min-max stats to df (in-place copy)."""
    df = df.copy()
    for col, s in stats.items():
        if col in df.columns:
            df[col] = (df[col] - s["min"]) / (s["max"] - s["min"] + 1e-8)
    return df


# ── Synthetic data generator ───────────────────────────────────────────────────
# Identical to demo_learning.py — kept inline so auto_retrain.py has no torch dep.

def generate_synthetic_data(n: int = N_SYNTHETIC, seed: int = 42) -> pd.DataFrame:
    """
    Generate n synthetic labelled training rows using domain-knowledge rules.
    Labels are deterministic given the seed; rules encode known risk factors.
    """
    rng  = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        defect_prob   = float(rng.beta(2, 5))
        cvss_sev      = int(rng.choice([0, 1, 2, 3], p=[0.30, 0.35, 0.25, 0.10]))
        conf_vals     = rng.dirichlet([1, 1, 1, 1]).tolist()
        conf_variance = float(np.std(conf_vals))
        cross_agree   = int(cvss_sev >= 2 and defect_prob >= 0.5)
        num_cwes      = int(min(rng.poisson(0.8), 9))
        cwe_flags     = {
            f"cwe_has_{cwe_name}": int(rng.random() < 0.15)
            for cwe_name in [
                "sql_injection", "xss", "buffer_overflow", "auth_bypass",
                "path_traversal", "improper_input", "use_after_free",
                "null_deref", "info_exposure",
            ]
        }
        atk_vec   = int(rng.choice([1, 2, 3], p=[0.30, 0.20, 0.50]))
        atk_cplx  = int(rng.choice([0, 1], p=[0.40, 0.60]))
        env       = int(rng.choice([0, 1, 2], p=[0.50, 0.30, 0.20]))
        svc_tier  = int(rng.choice([1, 2, 3], p=[0.20, 0.40, 0.40]))
        data_sens = int(rng.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.15, 0.10]))
        shadow    = int(rng.choice([-1, 0, 1], p=[0.30, 0.20, 0.50]))
        auth_touch = int(rng.random() < 0.20)
        db_touch   = int(rng.random() < 0.25)

        risk = (
            defect_prob * 30 + cvss_sev * 10 + num_cwes * 3
            + cwe_flags.get("cwe_has_sql_injection", 0) * 8
            + cwe_flags.get("cwe_has_auth_bypass", 0) * 10
            + (env == 2) * 15 + (svc_tier == 1) * 10
            + (data_sens >= 2) * 10 + auth_touch * 8
            + (shadow == 0) * 12 - (shadow == 1) * 10
        )
        risk     = float(np.clip(risk + rng.normal(0, 3), 0, 100))
        decision = 0 if risk < 30 else (1 if risk < 60 else 2)

        rows.append({
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
            "attack_vector_encoded":            atk_vec,
            "attack_complexity_encoded":        atk_cplx,
            "past_defects":                     int(rng.poisson(2)),
            "static_analysis_warnings":         int(rng.poisson(5)),
            "cyclomatic_complexity":            int(rng.integers(1, 30)),
            "test_coverage":                    float(rng.beta(5, 3)),
            "user_overrode_cvss":               int(rng.random() < 0.3),
            "user_cvss_override_direction":     int(rng.choice([-1, 0, 1])),
            "user_override_accuracy":           float(rng.beta(6, 3)),
            "user_response_time_seconds":       float(rng.exponential(45)),
            "shadow_twin_passed":               shadow,
            "user_feedback_sentiment":          float(rng.uniform(-0.3, 0.3)),
            "file_type_encoded":                int(rng.choice([0, 1, 2, 3, 4])),
            "diff_lines_added":                 int(rng.exponential(50)),
            "diff_lines_deleted":               int(rng.exponential(20)),
            "is_new_file":                      int(rng.random() < 0.15),
            "touches_auth_module":              auth_touch,
            "touches_db_layer":                 db_touch,
            "touches_api_boundary":             int(rng.random() < 0.30),
            "new_imports_count":                int(rng.poisson(1)),
            "deployment_environment":           env,
            "service_criticality_tier":         svc_tier,
            "data_sensitivity_level":           data_sens,
            "compliance_flags":                 int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7])),
            "branch_type":                      int(rng.choice([0, 1, 2])),
            "days_to_release_deadline":         int(rng.integers(1, 90)),
            "module_defect_rate_30d":           float(rng.beta(2, 10)),
            "developer_recent_defect_rate":     float(rng.beta(2, 10)),
            "instruction_mentions_security":    int(rng.random() < 0.25),
            "instruction_code_alignment_score": float(rng.beta(8, 2)),
            "session_codex_call_count":         int(rng.poisson(3) + 1),
            "consecutive_blocked_commits":      int(rng.poisson(0.5)),
            "risk_score":                       risk,
            "decision":                         decision,
        })
    return pd.DataFrame(rows)


# ── Dynamic-weighted synthetic generator ──────────────────────────────────────

def generate_weighted_synthetic_data(
    total_reviews: int,
    n: int = N_SYNTHETIC,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic training rows with feature weights that shift over time.

    Feature Tier evolution:
      Tier 1 (CVE + CWE + Defect ML outputs) — always high signal
      Tier 2 (human-in-the-loop: shadow_twin, user_feedback, user_overrode_cvss)
        → near-neutral at 0 reviews (data not yet collected)
        → full signal at 100+ reviews

    progress = min(total_reviews / 100, 1.0)
      0.0 → only ML pipeline features drive decisions
      1.0 → user feedback features carry equal weight

    This prevents the bootstrap model from learning spurious correlations on
    user-feedback features that are all-zero in early deployments.
    """
    progress = float(min(total_reviews / 100.0, 1.0))

    rng  = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        defect_prob   = float(rng.beta(2, 5))
        cvss_sev      = int(rng.choice([0, 1, 2, 3], p=[0.30, 0.35, 0.25, 0.10]))
        conf_vals     = rng.dirichlet([1, 1, 1, 1]).tolist()
        conf_variance = float(np.std(conf_vals))
        cross_agree   = int(cvss_sev >= 2 and defect_prob >= 0.5)
        num_cwes      = int(min(rng.poisson(0.8), 9))
        cwe_flags     = {
            f"cwe_has_{cwe_name}": int(rng.random() < 0.15)
            for cwe_name in [
                "sql_injection", "xss", "buffer_overflow", "auth_bypass",
                "path_traversal", "improper_input", "use_after_free",
                "null_deref", "info_exposure",
            ]
        }
        atk_vec   = int(rng.choice([1, 2, 3], p=[0.30, 0.20, 0.50]))
        atk_cplx  = int(rng.choice([0, 1], p=[0.40, 0.60]))
        env       = int(rng.choice([0, 1, 2], p=[0.50, 0.30, 0.20]))
        svc_tier  = int(rng.choice([1, 2, 3], p=[0.20, 0.40, 0.40]))
        data_sens = int(rng.choice([0, 1, 2, 3], p=[0.40, 0.35, 0.15, 0.10]))
        auth_touch = int(rng.random() < 0.20)
        db_touch   = int(rng.random() < 0.25)

        # Tier 2 features: near-neutral early, full signal as reviews accumulate
        # shadow_twin: probability of being non-zero scales with progress
        if rng.random() < progress:
            shadow = int(rng.choice([-1, 0, 1], p=[0.30, 0.20, 0.50]))
        else:
            shadow = 0  # not yet collected

        override_full  = int(rng.random() < 0.30)
        user_overrode  = int(rng.random() < (0.05 + 0.25 * progress))
        override_dir   = int(rng.choice([-1, 0, 1])) if user_overrode else 0
        user_acc_full  = float(rng.beta(6, 3))
        user_acc       = float(0.5 * (1 - progress) + user_acc_full * progress)
        resp_time      = float(rng.exponential(45))
        sentiment_full = float(rng.uniform(-0.3, 0.3))
        sentiment      = float(sentiment_full * progress)

        risk = (
            defect_prob * 30 + cvss_sev * 10 + num_cwes * 3
            + cwe_flags.get("cwe_has_sql_injection", 0) * 8
            + cwe_flags.get("cwe_has_auth_bypass", 0) * 10
            + (env == 2) * 15 + (svc_tier == 1) * 10
            + (data_sens >= 2) * 10 + auth_touch * 8
            + (shadow == 0) * 12 * progress   # shadow only influences risk once observed
            - (shadow == 1) * 10 * progress
        )
        risk     = float(np.clip(risk + rng.normal(0, 3), 0, 100))
        decision = 0 if risk < 30 else (1 if risk < 60 else 2)

        rows.append({
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
            "attack_vector_encoded":            atk_vec,
            "attack_complexity_encoded":        atk_cplx,
            "past_defects":                     int(rng.poisson(2)),
            "static_analysis_warnings":         int(rng.poisson(5)),
            "cyclomatic_complexity":            int(rng.integers(1, 30)),
            "test_coverage":                    float(rng.beta(5, 3)),
            "user_overrode_cvss":               user_overrode,
            "user_cvss_override_direction":     override_dir,
            "user_override_accuracy":           user_acc,
            "user_response_time_seconds":       resp_time,
            "shadow_twin_passed":               shadow,
            "user_feedback_sentiment":          sentiment,
            "file_type_encoded":                int(rng.choice([0, 1, 2, 3, 4])),
            "diff_lines_added":                 int(rng.exponential(50)),
            "diff_lines_deleted":               int(rng.exponential(20)),
            "is_new_file":                      int(rng.random() < 0.15),
            "touches_auth_module":              auth_touch,
            "touches_db_layer":                 db_touch,
            "touches_api_boundary":             int(rng.random() < 0.30),
            "new_imports_count":                int(rng.poisson(1)),
            "deployment_environment":           env,
            "service_criticality_tier":         svc_tier,
            "data_sensitivity_level":           data_sens,
            "compliance_flags":                 int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7])),
            "branch_type":                      int(rng.choice([0, 1, 2])),
            "days_to_release_deadline":         int(rng.integers(1, 90)),
            "module_defect_rate_30d":           float(rng.beta(2, 10)),
            "developer_recent_defect_rate":     float(rng.beta(2, 10)),
            "instruction_mentions_security":    int(rng.random() < 0.25),
            "instruction_code_alignment_score": float(rng.beta(8, 2)),
            "session_codex_call_count":         int(rng.poisson(3) + 1),
            "consecutive_blocked_commits":      int(rng.poisson(0.5)),
            "risk_score":                       risk,
            "decision":                         decision,
        })
    return pd.DataFrame(rows)


# ── State helpers ──────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if os.path.exists(STATE_JSON):
        with open(STATE_JSON) as f:
            return json.load(f)
    return {"reviews_since_retrain": 0, "total_reviews": 0, "history": []}


def _save_state(state: dict) -> None:
    with open(STATE_JSON, "w") as f:
        json.dump(state, f, indent=2)


# ── Public API ─────────────────────────────────────────────────────────────────

def run_retrain(verbose: bool = True) -> dict:
    """
    Retrain the DL meta-scorer on all accumulated reviews + synthetic data.

    Steps
    -----
    1. Load reviews.csv (all human-reviewed commits)
    2. Generate 5,000 synthetic rows
    3. Repeat real rows ×REAL_WEIGHT to amplify their signal
    4. Fit normalisation stats on combined data
    5. Split val = most recent 20% of real reviews
    6. Train MLPClassifier(128→64→32)
    7. Save model → dl_scorer_sklearn.pkl
    8. Save stats → norm_stats_sklearn.json
    9. Update retrain_state.json history

    Returns
    -------
    dict:
      val_accuracy    — float accuracy on held-out real reviews (nan if <5 rows)
      total_reviews   — int total real reviews consumed
      accuracy_delta  — float vs previous retrain (0.0 on first retrain)
    """
    if not os.path.exists(REVIEWS_CSV):
        raise FileNotFoundError(
            f"No reviews found at '{REVIEWS_CSV}'.\n"
            f"Use log_review() to accumulate human feedback first."
        )

    # ── Load real reviews ─────────────────────────────────────────────────────
    real_df       = pd.read_csv(REVIEWS_CSV)
    total_reviews = len(real_df)

    if verbose:
        dec_counts = real_df["decision"].value_counts().to_dict()
        print(f"\n{'='*60}")
        print(f"  AUTO-RETRAIN — DL Meta-Scorer")
        print(f"{'='*60}")
        print(f"  Real reviews loaded : {total_reviews}")
        print(f"  Decision breakdown  : "
              f"APPROVE={dec_counts.get(0,0)}  "
              f"FLAG={dec_counts.get(1,0)}  "
              f"BLOCK={dec_counts.get(2,0)}")

    # ── Val split: most recent 20% of real reviews ────────────────────────────
    val_size = max(1, int(total_reviews * 0.20))
    if total_reviews >= 5:
        val_df        = real_df.iloc[-val_size:].copy()
        train_real_df = real_df.iloc[:-val_size].copy()
    else:
        val_df        = pd.DataFrame(columns=real_df.columns)
        train_real_df = real_df.copy()

    if verbose:
        print(f"  Train real rows     : {len(train_real_df)}")
        print(f"  Val real rows       : {len(val_df)}  (most recent {val_size})")

    # ── Generate synthetic baseline (Tier 2 weight grows with total_reviews) ──
    progress_pct = min(int(total_reviews / 100 * 100), 100)
    if verbose:
        print(f"\n  Generating {N_SYNTHETIC} synthetic rows …")
        print(f"  Feature weighting   : Tier-1 (CVE+Defect) always-on  |  "
              f"Tier-2 (user feedback) at {progress_pct}% weight  "
              f"({total_reviews} reviews / 100 target)")
    synth_df = generate_weighted_synthetic_data(n=N_SYNTHETIC, total_reviews=total_reviews)

    # ── Combine: synthetic + (train_real × REAL_WEIGHT) ──────────────────────
    # Note: shadow_twin_passed (0/1) is used as a feature signal but we keep
    # equal weighting across all real rows — differential SHADOW_FAIL_WEIGHT
    # would create BLOCK bias with small datasets (e.g. 75:25 at R1).
    repeated_real = pd.concat([train_real_df] * REAL_WEIGHT, ignore_index=True)
    combined_df   = pd.concat([synth_df, repeated_real], ignore_index=True)

    if verbose:
        print(f"  Combined training   : {len(combined_df)} rows  "
              f"({N_SYNTHETIC} synthetic + {len(train_real_df)}×{REAL_WEIGHT} real)")

    # ── Fit normalisation ─────────────────────────────────────────────────────
    stats        = fit_normalize(combined_df)
    combined_norm = apply_normalize(combined_df, stats)
    X_train      = combined_norm[FEATURE_NAMES].values.astype(np.float32)
    y_train      = combined_norm["decision"].values.astype(int)

    # ── Train MLP ─────────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  Training MLP (128→64→32, early_stopping=True) …")

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=15,
        random_state=42,
        verbose=False,
    )
    model.fit(X_train, y_train)

    if verbose:
        print(f"  ✓ Training complete  (iterations: {model.n_iter_})")

    # ── Evaluate on val set ───────────────────────────────────────────────────
    val_accuracy = float("nan")
    if len(val_df) > 0:
        val_norm     = apply_normalize(val_df, stats)
        X_val        = val_norm[FEATURE_NAMES].values.astype(np.float32)
        y_val        = val_norm["decision"].values.astype(int)
        val_accuracy = float(accuracy_score(y_val, model.predict(X_val)))
        if verbose:
            n_correct = int(val_accuracy * len(val_df))
            print(f"  Val accuracy        : {val_accuracy*100:.1f}%  "
                  f"({n_correct}/{len(val_df)} correct)")
    else:
        if verbose:
            print(f"  Val accuracy        : — (fewer than 5 total reviews)")

    # ── Save model + stats ────────────────────────────────────────────────────
    joblib.dump(model, MODEL_OUT)
    with open(STATS_OUT, "w") as f:
        json.dump(stats, f, indent=2)

    if verbose:
        print(f"\n  ✓ Model saved  → {MODEL_OUT}")
        print(f"  ✓ Stats saved  → {STATS_OUT}")

    # ── Update retrain_state.json ─────────────────────────────────────────────
    state        = _load_state()
    history      = state.get("history", [])
    prev_acc     = history[-1].get("val_accuracy") if history else None
    prev_acc_f   = float(prev_acc) if prev_acc is not None else float("nan")

    if not np.isnan(val_accuracy) and not np.isnan(prev_acc_f):
        accuracy_delta = val_accuracy - prev_acc_f
    else:
        accuracy_delta = 0.0

    entry = {
        "retrain_count":      len(history) + 1,
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "total_real_reviews": total_reviews,
        "val_accuracy":       round(val_accuracy, 4) if not np.isnan(val_accuracy) else None,
        "accuracy_delta":     round(accuracy_delta, 4),
    }
    history.append(entry)
    state["history"]               = history
    state["reviews_since_retrain"] = 0   # reset trigger counter
    _save_state(state)

    if verbose:
        acc_str   = f"{val_accuracy*100:.1f}%" if not np.isnan(val_accuracy) else "N/A"
        delta_str = f"{accuracy_delta:+.4f}"
        print(f"\n  Retrain #{entry['retrain_count']}  "
              f"val_acc={acc_str}  delta={delta_str}")
        print(f"{'='*60}\n")

    return {
        "val_accuracy":   val_accuracy,
        "total_reviews":  total_reviews,
        "accuracy_delta": accuracy_delta,
    }


# ── CLI helpers ────────────────────────────────────────────────────────────────

def _print_status() -> None:
    """Print a formatted summary of retrain_state.json."""
    if not os.path.exists(STATE_JSON):
        print("No retrain history found.  Run 'python auto_retrain.py' after accumulating reviews.")
        return

    state   = _load_state()
    history = state.get("history", [])

    print(f"\n{'='*62}")
    print(f"  RETRAIN STATUS")
    print(f"{'='*62}")
    print(f"  Total reviews accumulated : {state.get('total_reviews', 0)}")
    print(f"  Reviews since last retrain: {state.get('reviews_since_retrain', 0)}")

    if not history:
        print("  No retrains have been run yet.")
    else:
        print(f"\n  {'#':>4}  {'Timestamp':<20}  {'Reviews':>8}  {'ValAcc':>7}  {'Delta':>8}")
        print("  " + "─" * 52)
        for h in history:
            ts    = h["timestamp"][:19].replace("T", " ")
            acc   = f"{h['val_accuracy']*100:.1f}%" if h["val_accuracy"] is not None else "    N/A"
            delta = f"{h['accuracy_delta']:+.4f}" if h["val_accuracy"] is not None else "      —"
            print(f"  {h['retrain_count']:>4}  {ts:<20}  {h['total_real_reviews']:>8}  {acc:>7}  {delta:>8}")

    print(f"{'='*62}\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-retrain the DL Meta-Scorer")
    parser.add_argument(
        "--status", action="store_true",
        help="Show retrain history from retrain_state.json and exit",
    )
    parser.add_argument(
        "--threshold", type=int, default=None,
        help="Only retrain if reviews_since_retrain >= THRESHOLD (else print status)",
    )
    args = parser.parse_args()

    if args.status:
        _print_status()

    elif args.threshold is not None:
        state = _load_state()
        since = state.get("reviews_since_retrain", 0)
        total = state.get("total_reviews", 0)
        print(f"Reviews since last retrain : {since}")
        print(f"Total reviews accumulated  : {total}")
        print(f"Threshold requested        : {args.threshold}")
        if since >= args.threshold:
            print(f"→ Threshold reached — retraining now …")
            run_retrain(verbose=True)
        else:
            print(f"→ Not yet ({args.threshold - since} more review(s) needed).")

    else:
        run_retrain(verbose=True)
