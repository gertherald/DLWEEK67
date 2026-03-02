"""
model.py — Wide & Deep MLP for the DL Meta-Scorer
==================================================
Architecture:
  Wide path  : linear(binary + ordinal inputs) → 64 hidden
  Deep path  : MLP [continuous inputs → 128 → 64 → 32]
  Embed path : embedding tables for high-cardinality categoricals → 32
  Fusion     : concat(wide=64, deep=32, embed=32) → 128 → 64
  Multi-task head:
    ├── risk_score  : 64 → 1  (sigmoid → ×100, regression 0–100)
    └── decision    : 64 → 3  (raw logits → APPROVE / FLAG_FOR_REVIEW / BLOCK)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_builder import FEATURE_NAMES

# ── Feature group definitions ─────────────────────────────────────────────────

# Categorical features that get learnable embedding tables
# name → number of distinct categories
EMBED_FEATURES: dict[str, int] = {
    "cvss_severity_encoded":     4,   # LOW/MEDIUM/HIGH/CRITICAL
    "file_type_encoded":         5,   # auth/db/api/frontend/infra
    "deployment_environment":    3,   # dev/staging/prod
    "service_criticality_tier":  3,   # tier 1/2/3 (mapped to 0/1/2)
    "data_sensitivity_level":    4,   # public/internal/PII/financial
    "branch_type":               3,   # feature/hotfix/main
}
EMBED_DIM = 4   # embedding dimension per category

# Binary / low-cardinality ordinal features → Wide path (linear shortcut)
WIDE_FEATURES: list[str] = [
    "cross_model_agreement",
    "cwe_has_sql_injection",
    "cwe_has_xss",
    "cwe_has_buffer_overflow",
    "cwe_has_auth_bypass",
    "cwe_has_path_traversal",
    "cwe_has_improper_input",
    "cwe_has_use_after_free",
    "cwe_has_null_deref",
    "cwe_has_info_exposure",
    "user_overrode_cvss",
    "user_cvss_override_direction",
    "shadow_twin_passed",
    "is_new_file",
    "touches_auth_module",
    "touches_db_layer",
    "touches_api_boundary",
    "instruction_mentions_security",
    "attack_complexity_encoded",
    "attack_vector_encoded",
    "compliance_flags",
    "num_cwes",
]

# Continuous / count features → Deep MLP path
DEEP_FEATURES: list[str] = [
    "defect_probability",
    "cvss_confidence_critical",
    "cvss_confidence_high",
    "cvss_confidence_medium",
    "cvss_confidence_low",
    "cvss_confidence_variance",
    "user_override_accuracy",
    "user_response_time_seconds",
    "user_feedback_sentiment",
    "test_coverage",
    "past_defects",
    "static_analysis_warnings",
    "cyclomatic_complexity",
    "diff_lines_added",
    "diff_lines_deleted",
    "new_imports_count",
    "days_to_release_deadline",
    "module_defect_rate_30d",
    "developer_recent_defect_rate",
    "instruction_code_alignment_score",
    "session_codex_call_count",
    "consecutive_blocked_commits",
]

# ── Index lookups for slicing the input tensor ────────────────────────────────
FEAT_IDX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}


def _idx(names: list[str]) -> list[int]:
    return [FEAT_IDX[n] for n in names]


WIDE_IDX:  list[int]      = _idx(WIDE_FEATURES)
DEEP_IDX:  list[int]      = _idx(DEEP_FEATURES)
EMBED_IDX: dict[str, int] = {name: FEAT_IDX[name] for name in EMBED_FEATURES}


# ── Model ─────────────────────────────────────────────────────────────────────

class WideDeepScorer(nn.Module):
    """
    Wide & Deep MLP for multi-task governance risk scoring.

    Input  : float tensor shape (batch, 50)
    Outputs:
      risk_score      — (batch,)   float 0–100
      decision_logits — (batch, 3) raw logits for APPROVE / FLAG_FOR_REVIEW / BLOCK
    """

    def __init__(
        self,
        wide_dim: int   = len(WIDE_FEATURES),
        deep_dim: int   = len(DEEP_FEATURES),
        dropout:  float = 0.3,
    ):
        super().__init__()

        # Embedding tables — one per categorical feature
        # padding_idx = n_cats so out-of-range values map to zero vector
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(n_cats + 1, EMBED_DIM, padding_idx=n_cats)
            for name, n_cats in EMBED_FEATURES.items()
        })
        embed_total = EMBED_DIM * len(EMBED_FEATURES)

        # Wide path: direct linear shortcut for binary/ordinal signals
        self.wide = nn.Linear(wide_dim, 64)

        # Deep path: 3-layer MLP for continuous signals
        self.deep = nn.Sequential(
            nn.Linear(deep_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Embedding projection: concat all embeddings → 32-d
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_total, 32),
            nn.ReLU(),
        )

        # Fusion: wide(64) + deep(32) + embed(32) = 128 → 64
        fusion_in = 64 + 32 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Multi-task heads
        self.head_risk     = nn.Linear(64, 1)   # sigmoid → ×100 → risk score
        self.head_decision = nn.Linear(64, 3)   # 3-class: APPROVE/FLAG/BLOCK

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Wide path
        w_out = F.relu(self.wide(x[:, WIDE_IDX]))

        # Deep path
        d_out = self.deep(x[:, DEEP_IDX])

        # Embedding path
        embed_parts = []
        for name, emb in self.embeddings.items():
            col = x[:, EMBED_IDX[name]].long().clamp(0, EMBED_FEATURES[name])
            embed_parts.append(emb(col))
        e_out = self.embed_proj(torch.cat(embed_parts, dim=-1))

        # Fusion
        fused = torch.cat([w_out, d_out, e_out], dim=-1)
        out   = self.fusion(fused)

        # Heads
        risk_score      = torch.sigmoid(self.head_risk(out)).squeeze(-1) * 100.0
        decision_logits = self.head_decision(out)

        return risk_score, decision_logits
