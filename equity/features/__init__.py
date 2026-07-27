"""
@module: equity.features
@depends: pandas, numpy
@exports: add_technical_features, apply_lags, build_features,
          FeatureConfig, DEFAULT_INDICATORS, NAIVE_INDICATOR_SPECS
@paper_ref: N/A
@data_flow: prices (+ optional sentiment_per_period) -> technical indicators ->
            sentiment LEFT-JOIN -> lag layer -> flat past-only feature matrix

S3 lag-aware technical + sentiment feature layer. Re-exports are LAZY so
``import equity.features`` is light (pandas/numpy loaded only when a builder
is actually called). Mirrors ``equity.sentiment`` lazy-export style.
"""

from __future__ import annotations


def __getattr__(name: str):  # pragma: no cover - thin re-export
    if name in (
        "add_technical_features",
        "FeatureConfig",
        "DEFAULT_INDICATORS",
        "NAIVE_INDICATOR_SPECS",
        "TECHNICAL_FEATURE_COLUMNS",
    ):
        from equity.features import technical

        return getattr(technical, name)
    if name in ("apply_lags", "LagConfig"):
        from equity.features import lag

        return getattr(lag, name)
    if name in ("build_features", "DEFAULT_LAG_WINDOWS"):
        from equity.features import build

        return getattr(build, name)
    raise AttributeError(f"module 'equity.features' has no attribute {name!r}")


__all__ = [
    "add_technical_features",
    "FeatureConfig",
    "DEFAULT_INDICATORS",
    "NAIVE_INDICATOR_SPECS",
    "TECHNICAL_FEATURE_COLUMNS",
    "apply_lags",
    "LagConfig",
    "DEFAULT_LAG_WINDOWS",
    "build_features",
]
