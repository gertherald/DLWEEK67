"""
scorer.py — Inference engine for the DL Meta-Scorer
====================================================
Loads the trained Wide & Deep model and produces:
  - risk_score    : 0–100 composite governance risk score
  - decision      : APPROVE / FLAG_FOR_REVIEW / BLOCK
  - top_factors   : ranked feature contributions (approximate SHAP)

Usage:
  python scorer.py --demo                      # high-risk SQL injection example
  python scorer.py --demo --model-path path/to/dl_scorer.pt

  Or import directly:
      from scorer import score_commit
      result = score_commit(cve_output, defect_output, user_signals, ...)
"""

import os
import json
import argparse
import numpy as np

from feature_builder import (
    build_feature_vector,
    feature_vector_to_array,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    DECISION_LABELS,
)

SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH         = os.path.join(SCRIPT_DIR, "dl_scorer.pt")
STATS_PATH         = os.path.join(SCRIPT_DIR, "norm_stats.json")
SKLEARN_MODEL_PATH = os.path.join(SCRIPT_DIR, "dl_scorer_sklearn.pkl")
SKLEARN_STATS_PATH = os.path.join(SCRIPT_DIR, "norm_stats_sklearn.json")

# Optional PyTorch — not required when sklearn fallback model is present
try:
    import torch
    from model import WideDeepScorer
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

DECISION_ICONS = {"APPROVE": "✅", "FLAG_FOR_REVIEW": "⚠️ ", "BLOCK": "🚫"}

_RISK_BANDS = [
    (30,  "🟢 LOW RISK"),
    (60,  "🟡 MEDIUM RISK"),
    (101, "🔴 HIGH RISK"),
]


# ── Model loading ─────────────────────────────────────────────────────────────

def _load_model(
    model_path: str = MODEL_PATH,
    stats_path: str = STATS_PATH,
) -> tuple:
    """
    Load model + normalisation stats.  Returns (model, stats, backend).

    Priority:
      1. PyTorch WideDeepScorer (dl_scorer.pt)       → backend="torch"
      2. sklearn MLPClassifier  (dl_scorer_sklearn.pkl) → backend="sklearn"
    """
    # ── Try PyTorch model ─────────────────────────────────────────────────────
    if _TORCH_AVAILABLE and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        model = WideDeepScorer()
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        stats = {}
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
        return model, stats, "torch"

    # ── Try sklearn fallback ──────────────────────────────────────────────────
    if os.path.exists(SKLEARN_MODEL_PATH):
        import joblib
        model = joblib.load(SKLEARN_MODEL_PATH)
        stats = {}
        if os.path.exists(SKLEARN_STATS_PATH):
            with open(SKLEARN_STATS_PATH) as f:
                stats = json.load(f)
        return model, stats, "sklearn"

    # ── Nothing available ─────────────────────────────────────────────────────
    hints = []
    if not _TORCH_AVAILABLE:
        hints.append("PyTorch not installed (pip install torch)")
    else:
        hints.append(f"PyTorch model not found at '{model_path}' — run: python train.py --demo")
    hints.append(
        f"Sklearn model not found at '{SKLEARN_MODEL_PATH}' — "
        f"run: python auto_retrain.py  (after logging some reviews)"
    )
    raise FileNotFoundError(
        "No trained model found.\n" + "\n".join(f"  • {h}" for h in hints)
    )


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalize(features: dict, stats: dict) -> dict:
    """Apply the same min-max normalisation used during training."""
    result = dict(features)
    for col, s in stats.items():
        if col in result:
            mn, mx = s["min"], s["max"]
            result[col] = (result[col] - mn) / (mx - mn + 1e-8)
    return result


# ── Approximate feature importance ────────────────────────────────────────────

def _top_factors(features: dict, top_n: int = 5) -> list[dict]:
    """
    Approximate feature contributions by how far each normalised value
    deviates from its neutral midpoint (0.5 on the 0–1 scale).

    For production, replace with shap.DeepExplainer(model, background_tensor)
    which provides exact gradient-based SHAP values.
    """
    factors = []
    for name, value in features.items():
        schema = FEATURE_SCHEMA.get(name)
        if schema is None:
            continue
        _, default, min_val, max_val, description = schema
        span = max_val - min_val
        if span == 0:
            continue
        normalised = (value - min_val) / span
        importance = abs(normalised - 0.5) * 2     # 0 = neutral, 1 = extreme
        direction  = "increases risk" if normalised > 0.5 else "decreases risk"
        factors.append({
            "feature":     name,
            "value":       value,
            "importance":  round(importance, 3),
            "direction":   direction,
            "description": description,
        })

    factors.sort(key=lambda x: x["importance"], reverse=True)
    return factors[:top_n]


# ── Main scoring function ─────────────────────────────────────────────────────

def score_commit(
    cve_output:    dict,
    defect_output: dict,
    user_signals:  dict | None = None,
    code_context:  dict | None = None,
    enterprise:    dict | None = None,
    instruction:   dict | None = None,
    model_path:    str         = MODEL_PATH,
    stats_path:    str         = STATS_PATH,
    verbose:       bool        = True,
) -> dict:
    """
    Full scoring pipeline: raw signals → feature vector → DL model → result.

    Parameters
    ----------
    cve_output    : output dict from CVE ML run_pipeline()
    defect_output : output dict from Defect ML predict
    user_signals  : user interaction signals (overrides, response time, etc.)
    code_context  : git diff metadata (file type, lines changed, etc.)
    enterprise    : organisational context (env, tier, sensitivity, etc.)
    instruction   : Codex call metadata (alignment score, session count, etc.)
    model_path    : path to dl_scorer.pt
    stats_path    : path to norm_stats.json
    verbose       : if True, print the formatted result to stdout

    Returns
    -------
    dict with keys:
      risk_score      — float 0–100
      risk_band       — '🟢 LOW RISK' | '🟡 MEDIUM RISK' | '🔴 HIGH RISK'
      decision        — 'APPROVE' | 'FLAG_FOR_REVIEW' | 'BLOCK'
      decision_probs  — {'APPROVE': float, 'FLAG_FOR_REVIEW': float, 'BLOCK': float}
      top_factors     — list of dicts (feature, value, importance, direction)
      features        — full 50-feature dict before normalisation
    """
    model, stats, backend = _load_model(model_path, stats_path)

    # Build raw feature vector
    features = build_feature_vector(
        cve_output    = cve_output,
        defect_output = defect_output,
        user_signals  = user_signals,
        code_context  = code_context,
        enterprise    = enterprise,
        instruction   = instruction,
    )

    # Normalise for inference
    normed = _normalize(features, stats)

    # ── Run model (PyTorch or sklearn) ────────────────────────────────────────
    if backend == "torch":
        x = torch.tensor(feature_vector_to_array(normed)).unsqueeze(0)
        with torch.no_grad():
            risk_tensor, dec_logits = model(x)
        risk_score   = float(risk_tensor[0])
        dec_probs    = torch.softmax(dec_logits[0], dim=0).numpy()
        decision_idx = int(dec_logits[0].argmax())

    else:  # sklearn MLPClassifier
        x            = feature_vector_to_array(normed).reshape(1, -1)
        dec_probs    = model.predict_proba(x)[0]   # shape (3,)
        decision_idx = int(model.predict(x)[0])
        # Derive a 0-100 risk score from probability-weighted class centres
        risk_score   = float(dec_probs[0] * 15 + dec_probs[1] * 50 + dec_probs[2] * 85)

    decision = DECISION_LABELS[decision_idx]
    decision_probs = {
        DECISION_LABELS[i]: round(float(dec_probs[i]), 3) for i in range(3)
    }

    risk_band = next(
        label for threshold, label in _RISK_BANDS if risk_score < threshold
    )

    top_factors = _top_factors(features)

    result = {
        "risk_score":     round(risk_score, 1),
        "risk_band":      risk_band,
        "decision":       decision,
        "decision_probs": decision_probs,
        "top_factors":    top_factors,
        "features":       features,
        "backend":        backend,
    }

    if verbose:
        _print_result(result)

    return result


# ── Formatted output ──────────────────────────────────────────────────────────

def _print_result(result: dict) -> None:
    icon    = DECISION_ICONS.get(result["decision"], "")
    backend = result.get("backend", "?")
    print(f"\n{'='*57}")
    print(f"  DL META-SCORER — GOVERNANCE RESULT  [{backend}]")
    print(f"{'='*57}")
    print(f"  Risk Score  :  {result['risk_score']:>5.1f} / 100   {result['risk_band']}")
    print(f"  Decision    :  {icon}  {result['decision']}")

    print(f"\n  Decision confidence:")
    for dec, prob in sorted(result["decision_probs"].items(), key=lambda x: x[1], reverse=True):
        bar   = "#" * int(prob * 30)
        d_ico = DECISION_ICONS.get(dec, "")
        print(f"    {d_ico}  {dec:<20} {prob*100:5.1f}%  [{bar:<30}]")

    print(f"\n  Top contributing factors:")
    for i, f in enumerate(result["top_factors"], 1):
        arrow = "↑" if f["direction"] == "increases risk" else "↓"
        print(f"    {i}. {arrow} {f['feature']:<42}= {f['value']}")
        print(f"         {f['description']}")

    print(f"{'='*57}\n")


# ── CLI / Demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL Meta-Scorer: score a commit")
    parser.add_argument("--demo",       action="store_true",
                        help="Run with a high-risk example (SQL injection to prod auth service)")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--stats-path", type=str, default=STATS_PATH)
    args = parser.parse_args()

    if args.demo:
        print("[Demo] High-risk commit: SQL injection code pushed to production auth service")

        cve_out = {
            "severity":   "CRITICAL",
            "confidence": {"CRITICAL": 0.75, "HIGH": 0.15, "MEDIUM": 0.07, "LOW": 0.03},
            "signals": {
                "cwe_has_sql_injection":  1, "cwe_has_auth_bypass":   1,
                "cwe_has_xss":            0, "cwe_has_buffer_overflow": 0,
                "cwe_has_path_traversal": 0, "cwe_has_improper_input":  1,
                "cwe_has_use_after_free": 0, "cwe_has_null_deref":      0,
                "cwe_has_info_exposure":  0,
                "num_cwes":              3,
                "attack_vector_encoded":      3,
                "attack_complexity_encoded":  1,
                "privileges_required_encoded": 2,
                "user_interaction_encoded":   0,
                "has_configurations": 0, "num_references": 0, "vuln_status_encoded": 0,
            },
        }

        defect_out = {
            "defect_probability": 0.82,
            "feature_values": {
                "past_defects":             5,
                "static_analysis_warnings": 23,
                "cyclomatic_complexity":    24,
                "test_coverage":            0.28,
                "response_for_class":       12,
            },
        }

        user_sig = {
            "user_overrode_cvss":            1,
            "user_cvss_override_direction":  -1,   # tried to downgrade
            "user_override_accuracy":        0.30,  # historically inaccurate
            "user_response_time_seconds":    4.2,   # reviewed in 4 seconds
            "shadow_twin_passed":            0,     # simulation failed
            "user_feedback_sentiment":       -0.2,
        }

        code_ctx = {
            "file_type":           "auth",
            "diff_lines_added":    85,
            "diff_lines_deleted":  12,
            "is_new_file":         0,
            "touches_auth_module": 1,
            "touches_db_layer":    1,
            "touches_api_boundary": 1,
            "new_imports_count":   2,
        }

        ent = {
            "deployment_environment":      "prod",
            "service_criticality_tier":    1,        # tier-1 (auth/payments)
            "data_sensitivity_level":      "PII",
            "compliance_flags":            3,         # PCI + HIPAA
            "branch_type":                 "hotfix",
            "days_to_release_deadline":    2,
            "module_defect_rate_30d":      0.18,
            "developer_recent_defect_rate": 0.22,
        }

        instr = {
            "instruction_mentions_security":    0,
            "instruction_code_alignment_score": 0.62,
            "session_codex_call_count":         12,
            "consecutive_blocked_commits":      2,
        }

        result = score_commit(
            cve_out, defect_out, user_sig, code_ctx, ent, instr,
            model_path = args.model_path,
            stats_path = args.stats_path,
        )

        # Print JSON summary (excluding full feature dict)
        summary = {k: v for k, v in result.items() if k != "features"}
        print(json.dumps(summary, indent=2))
    else:
        parser.print_help()
