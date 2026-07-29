"""
@module: equity.metrics
@depends: equity.metrics.accuracy
@exports: rmse, mae, directional_hit_rate, aggregate_accuracy
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §8 (S6 metrics)
@data_flow: prediction frames -> per-horizon accuracy / portfolio metrics

S6 evaluation metrics package. Lazy re-exports keep ``import equity.metrics`` light.
"""

from __future__ import annotations

__all__ = [
    "rmse",
    "mae",
    "directional_hit_rate",
    "aggregate_accuracy",
]


def __getattr__(name: str):  # pragma: no cover - thin re-export
    if name in {"rmse", "mae", "directional_hit_rate", "aggregate_accuracy"}:
        from equity.metrics import accuracy as _acc

        return getattr(_acc, name)
    raise AttributeError(f"module 'equity.metrics' has no attribute {name!r}")
