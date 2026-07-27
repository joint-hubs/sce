"""Equity diagnostics subpackage.

S1.3: ``published_at_guard`` leakage guard CLI. Re-exported lazily so
``import equity.diagnostics`` stays light (pandas/pandera loaded only when
the guard is actually run).
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
    raise AttributeError(f"module 'equity.diagnostics' has no attribute {name!r}")


__all__ = [
    "run_published_at_guard",
    "run_sentiment_aggregate_guard",
    "run_lookahead_indicator",
]
