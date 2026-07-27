"""
@module: equity.features.build
@depends: pandas, numpy, equity.features.technical, equity.features.lag
@exports: build_features, DEFAULT_LAG_WINDOWS
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §5.3 (S3 spec)
@data_flow: prices (+ optional sentiment_per_period) -> add_technical_features
            -> LEFT-JOIN sentiment (tz-canonicalize, fill NaN->0) -> apply_lags
            -> flat past-only feature matrix

S3 build orchestration (D4). Entry point :func:`build_features` returns a flat
feature matrix: one row per ``(ticker, period_close_ts)``, with all technical
indicators and lagged features appended. The matrix is usable independently
of SCE and the forecaster.

Timezone canonicalization (FOC-49 aggregator returns UTC; S1 prices are
America/New_York): both frames are converted to UTC before the LEFT-JOIN (mirrors
``equity/data/loader.py:943-949``). The output keeps prices' ``period_close_ts``
column (UTC-converted); sentiment rows whose ``period_close_ts`` matches
(after tz-conversion) are LEFT-JOINED. Missing ``(ticker, period)`` rows in the
sentiment frame (aggregator is a left-outer reduction -- missing = no articles)
are filled with NaN, then ``sentiment_*`` and ``n_articles`` NaN values are
filled with 0 (per D4: missing period = 0 articles, hence neutral score).

Lag layer is applied over BOTH technical AND sentiment base columns (D1).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from equity.features.lag import DEFAULT_LAG_WINDOWS, apply_lags
from equity.features.technical import (
    DEFAULT_INDICATORS,
    FeatureConfig,
    add_technical_features,
)

# Sentiment columns produced by FOC-49 aggregate_per_period (the LEFT-JOIN
# source). After the join + NaN fill, these become base columns for the lag
# layer (D1).
_SENTIMENT_BASE_COLS = (
    "sentiment_score",
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_neu",
    "n_articles",
)


def _canonicalize_tz_utc(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert ``df[col]`` to UTC in place (idempotent). Mirrors loader.py:943-949."""
    if col not in df.columns:
        return df
    out = df.copy()
    ts = out[col]
    if ts.dt.tz is None:
        # tz-naive -- assume UTC (the aggregator's canonical storage tz).
        out[col] = ts.dt.tz_localize("UTC")
    else:
        out[col] = ts.dt.tz_convert("UTC")
    return out


def build_features(
    prices: pd.DataFrame,
    sentiment_per_period: pd.DataFrame | None = None,
    *,
    lag_windows: Iterable[int] = DEFAULT_LAG_WINDOWS,
    indicator_cfg: FeatureConfig | None = None,
    lag_base_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build the flat past-only feature matrix.

    Pipeline (D4):

    1. :func:`add_technical_features` appends technical indicators to ``prices``.
    2. If ``sentiment_per_period`` is provided, both frames' ``period_close_ts``
       are canonicalized to UTC, then LEFT-JOINED on ``(ticker, period_close_ts)``.
       ``sentiment_*`` and ``n_articles`` NaN values are filled with 0
       (D4: missing period = 0 articles).
    3. :func:`apply_lags` is applied over the technical feature columns AND
       the sentiment base columns (D1). Lag windows default to
       :data:`DEFAULT_LAG_WINDOWS` = ``(1, 3, 5, 10, 21)``.

    The output is a flat long-form DataFrame, one row per
    ``(ticker, period_close_ts)``, with all feature columns appended.

    Parameters
    ----------
    prices:
        Canonical S1 prices frame (tz-aware ``period_close_ts``, America/New_York).
    sentiment_per_period:
        Optional FOC-49 aggregate frame (tz-aware UTC ``period_close_ts``).
    lag_windows:
        Lag windows ``L`` (default ``(1, 3, 5, 10, 21)``).
    indicator_cfg:
        Optional :class:`FeatureConfig` for the technical block.
    lag_base_cols:
        Optional explicit list of base columns to lag. Defaults to
        ``technical_feature_cols + sentiment_base_cols``. Use this to lag a
        subset (e.g. only sentiment) when needed.

    Returns
    -------
    pd.DataFrame
        Flat feature matrix. Does NOT mutate inputs.
    """
    cfg = indicator_cfg or DEFAULT_INDICATORS
    # 1. Technical indicators (past-only, per-ticker).
    feats = add_technical_features(prices, indicators=cfg)
    tech_cols = cfg.describe()

    # 2. LEFT-JOIN sentiment (tz-canonicalize, fill NaN -> 0 per D4).
    if sentiment_per_period is not None and not sentiment_per_period.empty:
        sent = sentiment_per_period.copy()
        feats = _canonicalize_tz_utc(feats, "period_close_ts")
        sent = _canonicalize_tz_utc(sent, "period_close_ts")
        # Drop any sentiment columns not in the canonical set; keep only the
        # join keys + _SENTIMENT_BASE_COLS.
        keep = ["ticker", "period_close_ts", *_SENTIMENT_BASE_COLS]
        sent = sent[[c for c in keep if c in sent.columns]]
        # LEFT-JOIN on (ticker, period_close_ts).
        feats = feats.merge(sent, on=["ticker", "period_close_ts"], how="left")
        # Fill NaN -> 0 for sentiment cols (D4). NaN here means "no articles
        # published for this (ticker, period)".
        for c in _SENTIMENT_BASE_COLS:
            if c in feats.columns:
                feats[c] = feats[c].fillna(0.0)
        sentiment_cols_present = [c for c in _SENTIMENT_BASE_COLS if c in feats.columns]
    elif sentiment_per_period is not None and sentiment_per_period.empty:
        # Empty sentiment frame: still LEFT-JOIN conceptually -- add zeroed
        # sentiment columns so downstream lag layer has stable input shape.
        for c in _SENTIMENT_BASE_COLS:
            feats[c] = 0.0
        sentiment_cols_present = list(_SENTIMENT_BASE_COLS)
    else:
        sentiment_cols_present = []

    # 3. Lag layer (D1): lag technical features AND sentiment base cols.
    if lag_base_cols is not None:
        base_cols = list(lag_base_cols)
    else:
        base_cols = list(tech_cols) + list(sentiment_cols_present)
    # Filter to columns actually present (defensive).
    base_cols = [c for c in base_cols if c in feats.columns]
    if base_cols:
        feats = apply_lags(feats, base_cols, windows=lag_windows)
    return feats


__all__ = ["build_features", "DEFAULT_LAG_WINDOWS"]
