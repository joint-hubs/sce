"""
@module: equity.forecaster
@depends: equity.forecaster.config, equity.forecaster.targets
@exports: HorizonConfig, SectorHeadForecaster, InstrumentResidualForecaster,
          QuantileHeadForecaster, add_forward_targets, run_smoke
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §7
@data_flow: enriched features + prices -> multi-horizon two-layer forecasts

S5 multi-horizon two-layer forecaster package (sector-head + instrument-residual
+ LightGBM quantile heads). Lazy re-exports keep ``import equity.forecaster``
light.
"""

from __future__ import annotations

__all__ = [
    "HorizonConfig",
    "SectorHeadParams",
    "ResidualHeadParams",
    "QuantileHeadParams",
    "SmokeConfig",
    "WalkForwardConfig",
    "add_forward_targets",
    "forward_target_col",
    "SectorHeadForecaster",
    "InstrumentResidualForecaster",
    "QuantileHeadForecaster",
    "run_smoke",
    "run_walk_forward",
    "write_metadata",
]


def __getattr__(name: str):  # pragma: no cover - thin re-export
    if name in {
        "HorizonConfig",
        "SectorHeadParams",
        "ResidualHeadParams",
        "QuantileHeadParams",
        "SmokeConfig",
        "WalkForwardConfig",
    }:
        from equity.forecaster import config as _cfg

        return getattr(_cfg, name)
    if name in {"add_forward_targets", "forward_target_col"}:
        from equity.forecaster import targets as _t

        return getattr(_t, name)
    if name == "SectorHeadForecaster":
        from equity.forecaster.sector_head import SectorHeadForecaster

        return SectorHeadForecaster
    if name == "InstrumentResidualForecaster":
        from equity.forecaster.residual import InstrumentResidualForecaster

        return InstrumentResidualForecaster
    if name == "QuantileHeadForecaster":
        from equity.forecaster.quantile import QuantileHeadForecaster

        return QuantileHeadForecaster
    if name == "run_smoke":
        from equity.forecaster.run_smoke import run_smoke

        return run_smoke
    if name == "run_walk_forward":
        from equity.forecaster.run_walk_forward import run_walk_forward

        return run_walk_forward
    if name == "write_metadata":
        from equity.forecaster.metadata import write_metadata

        return write_metadata
    raise AttributeError(f"module 'equity.forecaster' has no attribute {name!r}")
