"""
@module: equity.forecaster.config
@depends: dataclasses, typing
@exports: HorizonConfig, SectorHeadParams, ResidualHeadParams, QuantileHeadParams, SmokeConfig
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §7 (multi-horizon two-layer)
@data_flow: frozen knobs -> SectorHead / Residual / Quantile forecasters -> run_smoke

S5 forecaster config. Dataclass-only (no TOML section) — horizons, quantiles and
per-head tree hyperparameters have stable, locked defaults from GATE 1 / FOC-47
(Q5 horizons include 63; Q7 quantiles = 90% interval).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

DEFAULT_HORIZONS: Tuple[int, ...] = (1, 5, 10, 21, 63)
DEFAULT_QUANTILES: Tuple[float, ...] = (0.05, 0.5, 0.95)


@dataclass(frozen=True)
class HorizonConfig:
    """Locked multi-horizon + quantile knobs (FOC-52 S5)."""

    horizons: Tuple[int, ...] = DEFAULT_HORIZONS
    quantiles: Tuple[float, ...] = DEFAULT_QUANTILES
    time_col: str = "period_close_ts"
    ticker_col: str = "ticker"
    sector_col: str = "sector"
    n_folds: int = 5
    seed: int = 42


@dataclass(frozen=True)
class SectorHeadParams:
    """XGBoost params for the sector-head regressor (one model per horizon)."""

    n_estimators: int = 80
    max_depth: int = 4
    learning_rate: float = 0.08
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    n_jobs: int = 1
    verbosity: int = 0

    def to_xgb_kwargs(self, *, seed: int) -> Dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbosity,
            "random_state": seed,
            "enable_categorical": True,
            "tree_method": "hist",
        }


@dataclass(frozen=True)
class ResidualHeadParams:
    """XGBoost params for the instrument-residual regressor."""

    n_estimators: int = 80
    max_depth: int = 4
    learning_rate: float = 0.08
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_weight: float = 5.0
    reg_lambda: float = 1.0
    n_jobs: int = 1
    verbosity: int = 0

    def to_xgb_kwargs(self, *, seed: int) -> Dict[str, object]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "reg_lambda": self.reg_lambda,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbosity,
            "random_state": seed,
            "enable_categorical": True,
            "tree_method": "hist",
        }


@dataclass(frozen=True)
class QuantileHeadParams:
    """LightGBM quantile params (one model per (horizon, quantile))."""

    n_estimators: int = 80
    max_depth: int = 4
    learning_rate: float = 0.08
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    min_child_samples: int = 10
    reg_lambda: float = 1.0
    n_jobs: int = 1
    verbosity: int = -1

    def to_lgbm_kwargs(self, *, seed: int, alpha: float) -> Dict[str, object]:
        return {
            "objective": "quantile",
            "alpha": float(alpha),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_samples": self.min_child_samples,
            "reg_lambda": self.reg_lambda,
            "n_jobs": self.n_jobs,
            "verbosity": self.verbosity,
            "random_state": seed,
        }


@dataclass(frozen=True)
class SmokeConfig:
    """Single-fold train/test smoke runner knobs (S5.5)."""

    horizon: HorizonConfig = field(default_factory=HorizonConfig)
    sector: SectorHeadParams = field(default_factory=SectorHeadParams)
    residual: ResidualHeadParams = field(default_factory=ResidualHeadParams)
    quantile: QuantileHeadParams = field(default_factory=QuantileHeadParams)
    test_frac: float = 0.25
    run_grade: str = "exploratory"
    seed: int = 42
