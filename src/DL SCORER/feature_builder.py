"""
feature_builder.py — DL Meta-Scorer Feature Builder
====================================================
Aggregates signals from CVE ML, Defect ML, user interactions,
code change context, and enterprise governance into a single
50-feature vector for the Deep Learning meta-scorer.

Usage:
    from feature_builder import build_feature_vector, FEATURE_NAMES

    features = build_feature_vector(
        cve_output    = run_pipeline(...),   # from CVE ML
        defect_output = {...},               # from Defect ML
        user_signals  = {...},               # from interaction layer
        code_context  = {...},               # from git diff / file metadata
        enterprise    = {...},               # from org config
        instruction   = {...},              # from Codex call metadata
    )
"""

import numpy as np
from typing import Optional


# ── Feature schema ────────────────────────────────────────────────────────────
# name → (type, default, min, max, description)
FEATURE_SCHEMA: dict[str, tuple] = {

    # ── Tier 1: Automated ML outputs (features 1–24) ──────────────────────────
    "defect_probability":               ("float",  0.0, 0.0,   1.0,  "Defect ML probability of code defect"),
    "cvss_severity_encoded":            ("int",    0,   0,     3,    "CVSS severity: LOW=0 MEDIUM=1 HIGH=2 CRITICAL=3"),
    "cvss_confidence_critical":         ("float",  0.0, 0.0,   1.0,  "CVE ML confidence in CRITICAL label"),
    "cvss_confidence_high":             ("float",  0.0, 0.0,   1.0,  "CVE ML confidence in HIGH label"),
    "cvss_confidence_medium":           ("float",  0.0, 0.0,   1.0,  "CVE ML confidence in MEDIUM label"),
    "cvss_confidence_low":              ("float",  0.0, 0.0,   1.0,  "CVE ML confidence in LOW label"),
    "cvss_confidence_variance":         ("float",  0.0, 0.0,   1.0,  "Std-dev of 4 CVSS confidence scores — low = uncertain"),
    "cross_model_agreement":            ("binary", 0,   0,     1,    "1 if both ML models agree on HIGH/CRITICAL risk"),
    "num_cwes":                         ("int",    0,   0,     9,    "Total CWE flags triggered by code analysis"),
    "cwe_has_sql_injection":            ("binary", 0,   0,     1,    "SQL injection signal detected in generated code"),
    "cwe_has_xss":                      ("binary", 0,   0,     1,    "XSS signal detected in generated code"),
    "cwe_has_buffer_overflow":          ("binary", 0,   0,     1,    "Buffer overflow signal detected"),
    "cwe_has_auth_bypass":              ("binary", 0,   0,     1,    "Auth bypass signal detected"),
    "cwe_has_path_traversal":           ("binary", 0,   0,     1,    "Path traversal signal detected"),
    "cwe_has_improper_input":           ("binary", 0,   0,     1,    "Improper input validation detected"),
    "cwe_has_use_after_free":           ("binary", 0,   0,     1,    "Use-after-free signal detected"),
    "cwe_has_null_deref":               ("binary", 0,   0,     1,    "Null dereference signal detected"),
    "cwe_has_info_exposure":            ("binary", 0,   0,     1,    "Information exposure signal detected"),
    "attack_vector_encoded":            ("int",    1,   0,     3,    "Attack vector: PHYSICAL=0 LOCAL=1 ADJACENT=2 NETWORK=3"),
    "attack_complexity_encoded":        ("int",    1,   0,     1,    "Attack complexity: HIGH=0 LOW=1"),
    "past_defects":                     ("int",    0,   0,     100,  "Historical defect count (Defect ML top-5 feature)"),
    "static_analysis_warnings":         ("int",    0,   0,     500,  "Static analysis warning count"),
    "cyclomatic_complexity":            ("int",    1,   1,     200,  "Cyclomatic complexity of the function"),
    "test_coverage":                    ("float",  0.5, 0.0,   1.0,  "Test coverage fraction 0–1"),

    # ── Tier 2: Human-in-the-loop signals (features 25–30) ───────────────────
    "user_overrode_cvss":               ("binary", 0,   0,     1,    "1 if user changed ML severity label"),
    "user_cvss_override_direction":     ("int",    0,  -1,     1,    "Override direction: downgraded=-1 unchanged=0 upgraded=+1"),
    "user_override_accuracy":           ("float",  0.5, 0.0,   1.0,  "Rolling historical rate user overrides were correct"),
    "user_response_time_seconds":       ("float",  30.0,0.0,   3600.0,"Seconds user spent reviewing (very short = not reviewed)"),
    "shadow_twin_passed":               ("int",   -1,  -1,     1,    "Shadow simulation: -1=not run, 0=failed, 1=passed"),
    "user_feedback_sentiment":          ("float",  0.0,-1.0,   1.0,  "Sentiment of free-text feedback (-1 negative, +1 positive)"),

    # ── Tier 3: Code change context (features 31–38) ──────────────────────────
    "file_type_encoded":                ("int",    2,   0,     4,    "File type: auth=0 db=1 api=2 frontend=3 infra=4"),
    "diff_lines_added":                 ("int",    0,   0,     5000, "Lines added in this diff — larger = higher blast radius"),
    "diff_lines_deleted":               ("int",    0,   0,     5000, "Lines deleted — deletion of safety checks is high risk"),
    "is_new_file":                      ("binary", 0,   0,     1,    "1 if this is a brand-new file with no review history"),
    "touches_auth_module":              ("binary", 0,   0,     1,    "1 if changes touch auth/login code"),
    "touches_db_layer":                 ("binary", 0,   0,     1,    "1 if changes touch database layer (amplifies SQL risk)"),
    "touches_api_boundary":             ("binary", 0,   0,     1,    "1 if changes add/modify API endpoints"),
    "new_imports_count":                ("int",    0,   0,     50,   "Number of new package imports — each is a supply chain risk"),

    # ── Tier 4: Enterprise governance context (features 39–46) ───────────────
    "deployment_environment":           ("int",    0,   0,     2,    "Target env: dev=0 staging=1 prod=2"),
    "service_criticality_tier":         ("int",    3,   1,     3,    "Service tier: 1=payments/auth 2=core APIs 3=peripheral"),
    "data_sensitivity_level":           ("int",    0,   0,     3,    "Data class: public=0 internal=1 PII=2 financial=3"),
    "compliance_flags":                 ("int",    0,   0,     7,    "Bitmask: PCI=1 HIPAA=2 SOC2=4 (combinable)"),
    "branch_type":                      ("int",    0,   0,     2,    "Branch type: feature=0 hotfix=1 main=2"),
    "days_to_release_deadline":         ("int",    30,  0,     365,  "Days until next release — deadline pressure amplifies risk"),
    "module_defect_rate_30d":           ("float",  0.0, 0.0,   1.0,  "30-day rolling defect rate for this module"),
    "developer_recent_defect_rate":     ("float",  0.0, 0.0,   1.0,  "30-day rolling defect rate for this developer"),

    # ── Tier 5: Instruction/session quality signals (features 47–50) ──────────
    "instruction_mentions_security":    ("binary", 0,   0,     1,    "1 if instruction text mentions security"),
    "instruction_code_alignment_score": ("float",  0.8, 0.0,   1.0,  "Cosine similarity of instruction vs code embeddings"),
    "session_codex_call_count":         ("int",    1,   1,     200,  "Codex calls in this session — high count = rapid-fire"),
    "consecutive_blocked_commits":      ("int",    0,   0,     20,   "Consecutive blocked commits by this developer"),
}

# ── Retroactive incident labels (null at commit time, populated later) ─────────
# These are not model inputs — they become training targets once incidents are known.
INCIDENT_FEATURE_SCHEMA: dict[str, tuple] = {
    "caused_production_incident":  ("binary", None, 0,   1,    "1 if this commit was linked to a post-deploy incident"),
    "incident_severity":           ("int",    None, 0,   3,    "Severity of the incident: LOW=0 MEDIUM=1 HIGH=2 CRITICAL=3"),
    "incident_type":               ("int",    None, 0,   3,    "security=0 crash=1 data_loss=2 performance=3"),
    "days_until_incident":         ("int",    None, 0,   365,  "Days post-merge until incident was detected (nullable)"),
    "incident_linked_cwe":         ("int",    None, 0,   8,    "CWE index the incident traced back to (nullable)"),
}

# Ordered feature list — order MUST remain stable once training begins
FEATURE_NAMES: list[str] = list(FEATURE_SCHEMA.keys())

# ── Encoding maps ─────────────────────────────────────────────────────────────
SEVERITY_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
DECISION_LABELS = {0: "APPROVE", 1: "FLAG_FOR_REVIEW", 2: "BLOCK"}
DECISION_MAP    = {v: k for k, v in DECISION_LABELS.items()}
FILE_TYPE_MAP   = {"auth": 0, "db": 1, "api": 2, "frontend": 3, "infra": 4, "other": 2}
ENV_MAP         = {"dev": 0, "development": 0, "staging": 1, "prod": 2, "production": 2}
SENSITIVITY_MAP = {"public": 0, "internal": 1, "pii": 2, "financial": 3}
BRANCH_MAP      = {"feature": 0, "hotfix": 1, "main": 2, "master": 2, "release": 1}


# ── Helper: safe int-or-string decode ─────────────────────────────────────────
def _encode(raw, mapping: dict, default: int) -> int:
    if isinstance(raw, str):
        return mapping.get(raw.lower(), default)
    return int(raw) if raw is not None else default


# ── Feature builder ───────────────────────────────────────────────────────────

def build_feature_vector(
    cve_output:    dict,
    defect_output: dict,
    user_signals:  Optional[dict] = None,
    code_context:  Optional[dict] = None,
    enterprise:    Optional[dict] = None,
    instruction:   Optional[dict] = None,
) -> dict:
    """
    Aggregates signals from all sources into the 50-feature vector.

    Parameters
    ----------
    cve_output : dict
        Output from CVE ML run_pipeline():
        {
            'severity': 'HIGH',
            'confidence': {'LOW': 0.05, 'MEDIUM': 0.10, 'HIGH': 0.70, 'CRITICAL': 0.15},
            'signals': {cwe_flag: 0/1, attack_vector_encoded: int, ...}
        }

    defect_output : dict
        Output from Defect ML predict / single-instance:
        {
            'defect_probability': 0.72,
            'feature_values': {
                'past_defects': 3, 'static_analysis_warnings': 12,
                'cyclomatic_complexity': 18, 'test_coverage': 0.45,
            }
        }

    user_signals : dict, optional
        {
            'user_overrode_cvss': 0/1,
            'user_cvss_override_direction': -1/0/+1,
            'user_override_accuracy': float 0–1,
            'user_response_time_seconds': float,
            'shadow_twin_passed': -1/0/1,
            'user_feedback_sentiment': float -1–1,
        }

    code_context : dict, optional
        {
            'file_type': 'auth'|'db'|'api'|'frontend'|'infra'|'other',
            'diff_lines_added': int, 'diff_lines_deleted': int,
            'is_new_file': 0/1, 'touches_auth_module': 0/1,
            'touches_db_layer': 0/1, 'touches_api_boundary': 0/1,
            'new_imports_count': int,
        }

    enterprise : dict, optional
        {
            'deployment_environment': 'dev'|'staging'|'prod'  (or 0/1/2),
            'service_criticality_tier': 1/2/3,
            'data_sensitivity_level': 'public'|'internal'|'PII'|'financial' (or 0–3),
            'compliance_flags': int bitmask (PCI=1, HIPAA=2, SOC2=4),
            'branch_type': 'feature'|'hotfix'|'main' (or 0/1/2),
            'days_to_release_deadline': int,
            'module_defect_rate_30d': float,
            'developer_recent_defect_rate': float,
        }

    instruction : dict, optional
        {
            'instruction_mentions_security': 0/1,
            'instruction_code_alignment_score': float,
            'session_codex_call_count': int,
            'consecutive_blocked_commits': int,
        }

    Returns
    -------
    dict keyed by FEATURE_NAMES with all 50 scalar values.
    """
    user_signals = user_signals or {}
    code_context = code_context or {}
    enterprise   = enterprise   or {}
    instruction  = instruction  or {}

    # ── Parse CVE ML output ───────────────────────────────────────────────────
    severity_str  = cve_output.get("severity", "LOW")
    cvss_sev_enc  = SEVERITY_MAP.get(severity_str, 0)
    conf          = cve_output.get("confidence", {})
    conf_critical = float(conf.get("CRITICAL", 0.0))
    conf_high     = float(conf.get("HIGH",     0.0))
    conf_medium   = float(conf.get("MEDIUM",   0.0))
    conf_low      = float(conf.get("LOW",      0.0))
    conf_variance = float(np.std([conf_critical, conf_high, conf_medium, conf_low]))
    signals       = cve_output.get("signals", {})

    # ── Parse Defect ML output ────────────────────────────────────────────────
    defect_prob = float(defect_output.get("defect_probability", 0.0))
    feat_vals   = defect_output.get("feature_values", {})

    # ── Derived: cross-model agreement ────────────────────────────────────────
    cross_agree = int(cvss_sev_enc >= 2 and defect_prob >= 0.5)

    # ── Encoded categoricals ──────────────────────────────────────────────────
    file_type_enc = _encode(code_context.get("file_type", "other"), FILE_TYPE_MAP, 2)
    deploy_env    = _encode(enterprise.get("deployment_environment", "dev"), ENV_MAP, 0)
    data_sens     = _encode(enterprise.get("data_sensitivity_level", "public"), SENSITIVITY_MAP, 0)
    branch_enc    = _encode(enterprise.get("branch_type", "feature"), BRANCH_MAP, 0)

    # ── Assemble feature dict ─────────────────────────────────────────────────
    features = {
        # Tier 1 — Automated ML outputs
        "defect_probability":               defect_prob,
        "cvss_severity_encoded":            cvss_sev_enc,
        "cvss_confidence_critical":         conf_critical,
        "cvss_confidence_high":             conf_high,
        "cvss_confidence_medium":           conf_medium,
        "cvss_confidence_low":              conf_low,
        "cvss_confidence_variance":         conf_variance,
        "cross_model_agreement":            cross_agree,
        "num_cwes":                         int(signals.get("num_cwes", 0)),
        "cwe_has_sql_injection":            int(signals.get("cwe_has_sql_injection", 0)),
        "cwe_has_xss":                      int(signals.get("cwe_has_xss", 0)),
        "cwe_has_buffer_overflow":          int(signals.get("cwe_has_buffer_overflow", 0)),
        "cwe_has_auth_bypass":              int(signals.get("cwe_has_auth_bypass", 0)),
        "cwe_has_path_traversal":           int(signals.get("cwe_has_path_traversal", 0)),
        "cwe_has_improper_input":           int(signals.get("cwe_has_improper_input", 0)),
        "cwe_has_use_after_free":           int(signals.get("cwe_has_use_after_free", 0)),
        "cwe_has_null_deref":               int(signals.get("cwe_has_null_deref", 0)),
        "cwe_has_info_exposure":            int(signals.get("cwe_has_info_exposure", 0)),
        "attack_vector_encoded":            int(signals.get("attack_vector_encoded", 1)),
        "attack_complexity_encoded":        int(signals.get("attack_complexity_encoded", 1)),
        "past_defects":                     int(feat_vals.get("past_defects", 0)),
        "static_analysis_warnings":         int(feat_vals.get("static_analysis_warnings", 0)),
        "cyclomatic_complexity":            int(feat_vals.get("cyclomatic_complexity", 1)),
        "test_coverage":                    float(feat_vals.get("test_coverage", 0.5)),

        # Tier 2 — Human-in-the-loop signals
        "user_overrode_cvss":               int(user_signals.get("user_overrode_cvss", 0)),
        "user_cvss_override_direction":     int(user_signals.get("user_cvss_override_direction", 0)),
        "user_override_accuracy":           float(user_signals.get("user_override_accuracy", 0.5)),
        "user_response_time_seconds":       float(user_signals.get("user_response_time_seconds", 30.0)),
        "shadow_twin_passed":               int(user_signals.get("shadow_twin_passed", -1)),
        "user_feedback_sentiment":          float(user_signals.get("user_feedback_sentiment", 0.0)),

        # Tier 3 — Code change context
        "file_type_encoded":                file_type_enc,
        "diff_lines_added":                 int(code_context.get("diff_lines_added", 0)),
        "diff_lines_deleted":               int(code_context.get("diff_lines_deleted", 0)),
        "is_new_file":                      int(code_context.get("is_new_file", 0)),
        "touches_auth_module":              int(code_context.get("touches_auth_module", 0)),
        "touches_db_layer":                 int(code_context.get("touches_db_layer", 0)),
        "touches_api_boundary":             int(code_context.get("touches_api_boundary", 0)),
        "new_imports_count":                int(code_context.get("new_imports_count", 0)),

        # Tier 4 — Enterprise governance context
        "deployment_environment":           deploy_env,
        "service_criticality_tier":         int(enterprise.get("service_criticality_tier", 3)),
        "data_sensitivity_level":           data_sens,
        "compliance_flags":                 int(enterprise.get("compliance_flags", 0)),
        "branch_type":                      branch_enc,
        "days_to_release_deadline":         int(enterprise.get("days_to_release_deadline", 30)),
        "module_defect_rate_30d":           float(enterprise.get("module_defect_rate_30d", 0.0)),
        "developer_recent_defect_rate":     float(enterprise.get("developer_recent_defect_rate", 0.0)),

        # Tier 5 — Instruction/session quality
        "instruction_mentions_security":    int(instruction.get("instruction_mentions_security", 0)),
        "instruction_code_alignment_score": float(instruction.get("instruction_code_alignment_score", 0.8)),
        "session_codex_call_count":         int(instruction.get("session_codex_call_count", 1)),
        "consecutive_blocked_commits":      int(instruction.get("consecutive_blocked_commits", 0)),
    }

    return features


def feature_vector_to_array(features: dict) -> np.ndarray:
    """Convert feature dict to ordered numpy array using FEATURE_NAMES order."""
    return np.array([features[k] for k in FEATURE_NAMES], dtype=np.float32)


def get_defaults() -> dict:
    """Return all features set to their schema defaults."""
    return {name: schema[1] for name, schema in FEATURE_SCHEMA.items()}
