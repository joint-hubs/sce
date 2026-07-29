"""Equity diagnostics subpackage.

S1.3: ``published_at_guard`` leakage guard CLI. S3.3: ``lookahead_indicator``.
S4.5: ``walk_forward_monotonicity``, ``survivorship_check``. S4.6: SCE reuse
runners. S5.3: ``forward_target_isolation``. Re-exported lazily so
``import equity.diagnostics`` stays light (pandas/pandera/sklearn loaded only
when a symbol is actually accessed).
"""

from __future__ import annotations


def __getattr__(name: str):  # pragma: no cover - thin re-export
    if name == "run_published_at_guard":
        from equity.diagnostics.published_at_guard import run_published_at_guard

        return run_published_at_guard
    if name == "run_sentiment_aggregate_guard":
        from equity.diagnostics.sentiment_aggregate_guard import (
            run_sentiment_aggregate_guard,
        )

        return run_sentiment_aggregate_guard
    if name == "run_lookahead_indicator":
        from equity.diagnostics.lookahead_indicator import run_lookahead_indicator

        return run_lookahead_indicator
    if name == "run_walk_forward_monotonicity":
        from equity.diagnostics.walk_forward_monotonicity import (
            run_walk_forward_monotonicity,
        )

        return run_walk_forward_monotonicity
    if name == "run_survivorship_check":
        from equity.diagnostics.survivorship_check import run_survivorship_check

        return run_survivorship_check
    if name == "run_forward_target_isolation":
        from equity.diagnostics.forward_target_isolation import (
            run_forward_target_isolation,
        )

        return run_forward_target_isolation
    if name in {
        "evaluate_equity_sce",
        "run_permuted_target_equity",
        "run_shuffled_groups_equity",
        "run_crossfit_ab_equity",
        "audit_feature_dominance_equity",
    }:
        from equity.diagnostics import sce_reuse as _sce_reuse

        return getattr(_sce_reuse, name)
    raise AttributeError(f"module 'equity.diagnostics' has no attribute {name!r}")


__all__ = [
    "run_published_at_guard",
    "run_sentiment_aggregate_guard",
    "run_lookahead_indicator",
    "run_walk_forward_monotonicity",
    "run_survivorship_check",
    "run_forward_target_isolation",
    "evaluate_equity_sce",
    "run_permuted_target_equity",
    "run_shuffled_groups_equity",
    "run_crossfit_ab_equity",
    "audit_feature_dominance_equity",
]
