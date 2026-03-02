"""
review_logger.py — Log a human review decision and trigger auto-retrain
========================================================================
Single entry point called after each human review of a Codex-generated
code commit.  Appends one row to reviews.csv and triggers a full retrain
every RETRAIN_THRESHOLD reviews.

RETRAIN_THRESHOLD defaults to 20 (good for MVP: ~2-day retrain cadence
at 10 reviews/day).  Bump to 50 once 200+ total reviews are accumulated.

Usage:
    from review_logger import log_review

    result = log_review(
        cve_output     = { ... },    # from CVE ML run_pipeline()
        defect_output  = { ... },    # from Defect ML predict_defect()
        human_decision = "BLOCK",    # "APPROVE" | "FLAG_FOR_REVIEW" | "BLOCK"
    )
    print(result)
    # {'logged': True, 'retrain_triggered': False, 'reviews_until_retrain': 15, 'total_reviews': 5}

    # After 20 reviews:
    # {'logged': True, 'retrain_triggered': True, 'reviews_until_retrain': 0,
    #  'total_reviews': 20, 'new_accuracy': 0.75}
"""

import os
import sys
import csv
import json
import warnings

warnings.filterwarnings("ignore")

# ── Configuration ─────────────────────────────────────────────────────────────
RETRAIN_THRESHOLD = 20   # trigger a retrain every N reviews
                         # → bump to 50 once 200+ total reviews exist

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REVIEWS_CSV = os.path.join(SCRIPT_DIR, "reviews.csv")
STATE_JSON  = os.path.join(SCRIPT_DIR, "retrain_state.json")

sys.path.insert(0, SCRIPT_DIR)
from feature_builder import (       # noqa: E402
    build_feature_vector,
    FEATURE_NAMES,
    DECISION_MAP,
    DECISION_LABELS,
)


# ── Risk score helper ──────────────────────────────────────────────────────────

def _compute_risk_score(features: dict, human_decision: str) -> float:
    """
    Derive a 0–100 risk_score from the human decision + ML signals.

    Base scores by class:
      APPROVE → 15   FLAG_FOR_REVIEW → 45   BLOCK → 80

    Adjustments:
      + num_cwes × 3
      + defect_probability × 20
    """
    base = {"APPROVE": 15, "FLAG_FOR_REVIEW": 45, "BLOCK": 80}.get(human_decision, 45)
    adj  = (
        features.get("num_cwes", 0) * 3
        + features.get("defect_probability", 0.0) * 20
    )
    return float(min(max(base + adj, 0), 100))


# ── State helpers ──────────────────────────────────────────────────────────────

def _load_state() -> dict:
    if os.path.exists(STATE_JSON):
        with open(STATE_JSON) as f:
            return json.load(f)
    return {"reviews_since_retrain": 0, "total_reviews": 0, "history": []}


def _save_state(state: dict) -> None:
    with open(STATE_JSON, "w") as f:
        json.dump(state, f, indent=2)


# ── CSV append ─────────────────────────────────────────────────────────────────

def _append_to_csv(features: dict, decision_int: int, risk_score: float) -> None:
    """Append one 52-column row to reviews.csv (creates file + header if new)."""
    columns      = FEATURE_NAMES + ["decision", "risk_score"]
    row_values   = [features[k] for k in FEATURE_NAMES] + [decision_int, round(risk_score, 2)]
    write_header = not os.path.exists(REVIEWS_CSV)

    with open(REVIEWS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(columns)
        writer.writerow(row_values)


# ── Public API ─────────────────────────────────────────────────────────────────

def log_review(
    cve_output:     dict,
    defect_output:  dict,
    user_signals:   dict | None = None,
    code_context:   dict | None = None,
    enterprise:     dict | None = None,
    instruction:    dict | None = None,
    human_decision: str = "FLAG_FOR_REVIEW",
) -> dict:
    """
    Log one human-reviewed commit and trigger auto-retrain when threshold reached.

    Parameters
    ----------
    cve_output     : output dict from CVE ML run_pipeline()
    defect_output  : output dict from Defect ML predict_defect()
    user_signals   : optional human-in-the-loop signals
    code_context   : optional git diff / file metadata
    enterprise     : optional org governance context
    instruction    : optional Codex call metadata
    human_decision : "APPROVE" | "FLAG_FOR_REVIEW" | "BLOCK"

    Returns
    -------
    dict:
      logged                — True
      total_reviews         — cumulative review count
      retrain_triggered     — True if retrain fired this call
      reviews_until_retrain — int (0 if retrain fired)
      new_accuracy          — float (only when retrain_triggered=True)
    """
    human_decision = human_decision.upper().strip()
    if human_decision not in DECISION_MAP:
        raise ValueError(
            f"human_decision must be one of {list(DECISION_MAP)}; got '{human_decision}'"
        )

    # ── Build 50-feature vector ───────────────────────────────────────────────
    features     = build_feature_vector(
        cve_output    = cve_output,
        defect_output = defect_output,
        user_signals  = user_signals,
        code_context  = code_context,
        enterprise    = enterprise,
        instruction   = instruction,
    )
    decision_int = DECISION_MAP[human_decision]
    risk_score   = _compute_risk_score(features, human_decision)

    # ── Persist ───────────────────────────────────────────────────────────────
    _append_to_csv(features, decision_int, risk_score)

    state = _load_state()
    state["reviews_since_retrain"] = state.get("reviews_since_retrain", 0) + 1
    state["total_reviews"]         = state.get("total_reviews", 0) + 1
    _save_state(state)

    total = state["total_reviews"]
    since = state["reviews_since_retrain"]

    # ── Trigger retrain? ──────────────────────────────────────────────────────
    if since >= RETRAIN_THRESHOLD:
        from auto_retrain import run_retrain   # lazy import — avoids circular dep
        retrain_result = run_retrain(verbose=False)
        # run_retrain() resets reviews_since_retrain → 0 in STATE_JSON
        return {
            "logged":                True,
            "total_reviews":         total,
            "retrain_triggered":     True,
            "reviews_until_retrain": 0,
            "new_accuracy":          retrain_result["val_accuracy"],
        }

    return {
        "logged":                True,
        "total_reviews":         total,
        "retrain_triggered":     False,
        "reviews_until_retrain": RETRAIN_THRESHOLD - since,
    }


# ── Quick self-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, shutil, math

    print("Self-test: log 3 reviews and verify state …")

    # Back up any existing state / CSV
    _backup_csv   = REVIEWS_CSV   + ".bak"
    _backup_state = STATE_JSON    + ".bak"
    for src, dst in [(REVIEWS_CSV, _backup_csv), (STATE_JSON, _backup_state)]:
        if os.path.exists(src):
            shutil.copy(src, dst)
            os.remove(src)

    try:
        _cve = {
            "severity": "MEDIUM",
            "confidence": {"CRITICAL": 0.05, "HIGH": 0.20, "MEDIUM": 0.60, "LOW": 0.15},
            "signals": {
                "num_cwes": 1, "cwe_has_sql_injection": 0, "cwe_has_xss": 1,
                "cwe_has_improper_input": 0, "attack_vector_encoded": 3,
                "attack_complexity_encoded": 1, "cwe_has_buffer_overflow": 0,
                "cwe_has_path_traversal": 0, "cwe_has_use_after_free": 0,
                "cwe_has_null_deref": 0, "cwe_has_auth_bypass": 0,
                "cwe_has_info_exposure": 0,
            },
        }
        _defect = {
            "defect_probability": 0.35,
            "feature_values": {
                "past_defects": 1, "static_analysis_warnings": 3,
                "cyclomatic_complexity": 8, "test_coverage": 0.6,
                "response_for_class": 5,
            },
        }

        for i in range(3):
            r = log_review(_cve, _defect, human_decision="FLAG_FOR_REVIEW")
            triggered = r["retrain_triggered"]
            remaining = r["reviews_until_retrain"]
            print(f"  Review {i+1}: logged={r['logged']}  "
                  f"triggered={triggered}  until_retrain={remaining}  "
                  f"total={r['total_reviews']}")

        import pandas as pd
        df = pd.read_csv(REVIEWS_CSV)
        print(f"\n  reviews.csv: {len(df)} rows × {len(df.columns)} columns  ✓")
        assert len(df) == 3, "Expected 3 rows"
        assert "decision" in df.columns and "risk_score" in df.columns
        print("  All assertions passed.")

    finally:
        # Restore backups
        os.remove(REVIEWS_CSV) if os.path.exists(REVIEWS_CSV) else None
        os.remove(STATE_JSON)  if os.path.exists(STATE_JSON)  else None
        for bak, orig in [(_backup_csv, REVIEWS_CSV), (_backup_state, STATE_JSON)]:
            if os.path.exists(bak):
                shutil.copy(bak, orig)
                os.remove(bak)
        print("  State restored.")
