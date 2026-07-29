"""
@module: equity.metrics
@depends: equity.metrics.accuracy, equity.metrics.sharpe
@exports: rmse, mae, directional_hit_rate, aggregate_accuracy,
          decile_long_short_returns, sharpe_ratio, sortino_ratio,
          select_horizon, aggregate_portfolio
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
    "decile_long_short_returns",
    "sharpe_ratio",
    "sortino_ratio",
    "select_horizon",
    "aggregate_portfolio",
]

_ACC = {"rmse", "mae", "directional_hit_rate", "aggregate_accuracy"}
_SHARPE = {
    "decile_long_short_returns",
    "sharpe_ratio",
    "sortino_ratio",
    "select_horizon",
    "aggregate_portfolio",
}


def __getattr__(name: str):  # pragma: no cover - thin re-export
    if name in _ACC:
        from equity.metrics import accuracy as _acc

        return getattr(_acc, name)
    if name in _SHARPE:
        from equity.metrics import sharpe as _sh

        return getattr(_sh, name)
    raise AttributeError(f"module 'equity.metrics' has no attribute {name!r}")
