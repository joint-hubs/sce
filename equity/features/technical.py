"""
@module: equity.features.technical
@depends: pandas, numpy
@exports: add_technical_features, FeatureConfig, DEFAULT_INDICATORS,
          NAIVE_INDICATOR_SPECS, TECHNICAL_FEATURE_COLUMNS
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §5.3 (S3 spec)
@data_flow: prices -> per-ticker past-only technical indicators appended in place

S3.1 technical indicators. EVERY indicator is past-only (FOOTGUN #1, PRD §6.4.1):
for any feature column ``f`` and row at time ``t``, ``f(t)`` is a function of
``prices[:t]`` only (rows strictly before ``t`` within the same ticker group).
Concretely: each naive (current-row-inclusive) indicator is shifted by 1, so
the value stored at row ``t`` reflects data through ``close[t-1]``.

Per-ticker independence (D7): every rolling op is applied via
``groupby("ticker", sort=False)`` so a window never bleeds across the ticker
boundary. Inputs must be sorted ascending by ``period_close_ts`` within ticker;
the multi-ticker entry points sort defensively.

Generic recompute core (D3): ``NAIVE_INDICATOR_SPECS`` maps each feature column
to a per-ticker single-ticker spec function that returns the NAIVE (unshifted)
indicator series. The lookahead guard
(:mod:`equity.diagnostics.lookahead_indicator`) feeds each ticker's
``prices[:t]`` slice through the matching spec fn and compares the last value
to the stored feature -- this is the exact leak the guard catches. New
indicators (and sentiment roll features, later) are added by extending
``NAIVE_INDICATOR_SPECS`` -- no guard rewrite needed.

NaN in early windows (D5): pandas rolling/ewm emit native NaN during warmup;
we do NOT mask. The guard treats NaN as "undefined at this row", NOT a
violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

# Canonical price columns consumed from the S1 prices frame (see
# equity.data.schema.CANONICAL_PRICE_COLUMNS). We do NOT re-validate here --
# the loader already did. We only READ these columns.
_PRICE_COLS = ("open", "high", "low", "close", "adj_close", "volume", "hlc_average")

# Past-only convention: every naive indicator is shifted by 1 so the feature
# at row t uses only rows strictly before t (FOOTGUN #1, PRD §6.4.1).
_PAST_SHIFT = 1


# ---------------------------------------------------------------------------
# Per-ticker NAIVE indicator helpers (current-row-inclusive). The multi-ticker
# add_* functions apply these per group then .shift(_PAST_SHIFT) to produce
# the past-only feature. The same helpers are reused by the lookahead guard.
# ---------------------------------------------------------------------------


def _sma_naive(close: pd.Series, window: int) -> pd.Series:
    """Simple moving mean of close, current-row-INCLUSIVE (no shift).

    Uses pandas default ``closed='right'`` (includes the current row) -- this
    is the NAIVE form. The past-only feature is ``_sma_naive(close, w).shift(1)``.
    """
    return close.rolling(window, closed="right").mean()


def _ema_naive(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving mean of close, current-row-inclusive."""
    return close.ewm(span=span, adjust=False).mean()


def _logret_naive(close: pd.Series) -> pd.Series:
    """log(close/close.shift(1)) -- current-row-inclusive (uses close[t])."""
    return np.log(close / close.shift(1))


def _ret_n_naive(close: pd.Series, n: int) -> pd.Series:
    """log(close[t]/close[t-n]) -- current-row-inclusive."""
    return np.log(close / close.shift(n))


def _rsi_naive(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI(period), current-row-inclusive.

    Uses ``ewm(alpha=1/period, adjust=False)`` which is the Wilder smoothing
    equivalent to the classic ``SMMA = (prev_avg*(period-1) + cur) / period``.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    # Losses are the negative deltas, taken as positive magnitudes.
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # When avg_loss == 0 (no losses in window) RSI = 100; when avg_gain == 0
    # RSI = 0. pandas produces NaN/inf here -- canonicalize per Wilder.
    # ``Series.where(cond, other)`` keeps the original where cond is True and
    # substitutes ``other`` where False -- so we keep rsi where avg_loss != 0
    # and substitute 100 where avg_loss == 0.
    rsi = rsi.where(avg_loss != 0, other=100.0)
    rsi = rsi.where(avg_gain != 0, other=0.0)
    return rsi


def _macd_naive(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line + signal + histogram, current-row-inclusive. Returns a 3-col frame."""
    ema_fast = _ema_naive(close, fast)
    ema_slow = _ema_naive(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist},
        index=close.index,
    )


def _volatility_naive(close: pd.Series, window: int) -> pd.Series:
    """Rolling std of log returns over window, current-row-inclusive.

    logret[t] uses close[t] and close[t-1]; the naive std at t therefore
    includes close[t]. The past-only feature shifts by 1.
    """
    logret = _logret_naive(close)
    return logret.rolling(window, closed="right").std()


def _volume_zscore_naive(volume: pd.Series, window: int = 21) -> pd.Series:
    """Z-score of volume vs its own rolling mean/std, current-row-inclusive.

    Naive form: ``(volume[t] - mean(vol[t-w..t])) / std(vol[t-w..t])``. Past-only
    feature shifts by 1 so the value at row t reflects volume[t-1] and stats
    of ``volume[t-w..t-1]``.
    """
    mean = volume.rolling(window, closed="right").mean()
    std = volume.rolling(window, closed="right").std()
    return (volume - mean) / std


def _atr_naive(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (Wilder), current-row-inclusive.

    TR[t] = max(high[t]-low[t], |high[t]-close[t-1]|, |low[t]-close[t-1]|). This
    uses close[t-1] (past) AND high[t]/low[t] (current) -- the naive form is
    current-row-inclusive. Past-only feature shifts by 1.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    return atr


def _bollinger_naive(close: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Bollinger mid/upper/lower, current-row-inclusive."""
    mid = close.rolling(window, closed="right").mean()
    std = close.rolling(window, closed="right").std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return pd.DataFrame(
        {"bb_mid": mid, "bb_upper": upper, "bb_lower": lower},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Multi-ticker add_* entry points. Each applies the naive per-ticker helper
# via groupby.transform, then .shift(_PAST_SHIFT) WITHIN the group to produce
# the past-only feature. groupby(sort=False) preserves caller order; we sort
# defensively on period_close_ts at the top of add_technical_features.
# ---------------------------------------------------------------------------


def _per_ticker_shifted(
    prices: pd.DataFrame, col: str, fn: Callable[[pd.Series], pd.Series]
) -> pd.Series:
    """Apply ``fn`` per ticker to ``prices[col]`` and shift the result by
    ``_PAST_SHIFT`` within each ticker group. Returns a Series aligned to
    ``prices.index`` (NaN in warmup, unmodified by D5).

    The naive fn runs INSIDE the per-group transform, then ``.shift(1)`` is
    applied to the naive output -- so row ``t`` stores the naive value at
    ``t-1`` (past-only, FOOTGUN #1).
    """
    grouped = prices.groupby("ticker", sort=False)[col]
    return grouped.transform(lambda s: fn(s).shift(_PAST_SHIFT))


def _per_ticker_multi_col(
    prices: pd.DataFrame, fn: Callable[[pd.DataFrame], pd.DataFrame]
) -> pd.DataFrame:
    """Apply a multi-output naive fn per ticker (returns a DataFrame) and shift
    each output by ``_PAST_SHIFT`` within the ticker group. Returns a frame
    aligned to ``prices.index``.
    """
    # groupby.apply on a function returning a DataFrame; pandas concatenates
    # and reindexes to original order. Then shift within group.
    parts: list[pd.DataFrame] = []
    for _, g in prices.groupby("ticker", sort=False):
        naive = fn(g)
        # Shift each output column within this single-ticker group.
        shifted = naive.shift(_PAST_SHIFT)
        shifted.index = g.index  # realign to original positions
        parts.append(shifted)
    if parts:
        return pd.concat(parts).reindex(prices.index)
    return pd.DataFrame(index=prices.index)


def add_returns(prices: pd.DataFrame, *, horizons=(1, 5, 10, 21)) -> pd.DataFrame:
    """Append log-return features ``ret_{N}d_log`` for each horizon, past-only.

    ``ret_{N}d_log[t] = log(close[t-1] / close[t-1-N])`` -- uses only rows
    strictly before ``t``.
    """
    out = prices
    for n in horizons:
        col = f"ret_{n}d_log"
        out = out.assign(**{col: _per_ticker_shifted(out, "close", lambda s: _ret_n_naive(s, n))})
    return out


def add_sma(prices: pd.DataFrame, *, windows=(5, 10, 21, 63)) -> pd.DataFrame:
    """Append ``sma_{N}`` columns (past-only)."""
    out = prices
    for w in windows:
        col = f"sma_{w}"
        out = out.assign(
            **{col: _per_ticker_shifted(out, "close", lambda s, w=w: _sma_naive(s, w))}
        )
    return out


def add_ema(prices: pd.DataFrame, *, windows=(5, 10, 21, 63)) -> pd.DataFrame:
    """Append ``ema_{N}`` columns (past-only)."""
    out = prices
    for w in windows:
        col = f"ema_{w}"
        out = out.assign(
            **{col: _per_ticker_shifted(out, "close", lambda s, w=w: _ema_naive(s, w))}
        )
    return out


def add_rsi(prices: pd.DataFrame, *, period: int = 14) -> pd.DataFrame:
    """Append ``rsi_{period}`` (past-only Wilder's RSI)."""
    col = f"rsi_{period}"
    return prices.assign(
        **{col: _per_ticker_shifted(prices, "close", lambda s, p=period: _rsi_naive(s, p))}
    )


def add_macd(
    prices: pd.DataFrame, *, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Append ``macd``, ``macd_signal``, ``macd_hist`` (past-only)."""
    out = prices.copy()
    naive = _per_ticker_multi_col(out, lambda g: _macd_naive(g["close"], fast, slow, signal))
    for col in ("macd", "macd_signal", "macd_hist"):
        out[col] = naive[col]
    return out


def add_volatility(prices: pd.DataFrame, *, windows=(21, 63)) -> pd.DataFrame:
    """Append ``volatility_{N}`` (rolling std of log returns, past-only)."""
    out = prices
    for w in windows:
        col = f"volatility_{w}"
        out = out.assign(
            **{col: _per_ticker_shifted(out, "close", lambda s, w=w: _volatility_naive(s, w))}
        )
    return out


def add_volume_zscore(prices: pd.DataFrame, *, window: int = 21) -> pd.DataFrame:
    """Append ``volume_zscore_{window}`` (past-only)."""
    col = f"volume_zscore_{window}"
    return prices.assign(
        **{
            col: _per_ticker_shifted(
                prices, "volume", lambda s, w=window: _volume_zscore_naive(s, w)
            )
        }
    )


def add_atr(prices: pd.DataFrame, *, period: int = 14) -> pd.DataFrame:
    """Append ``atr_{period}`` (past-only Wilder ATR)."""
    col = f"atr_{period}"
    # Multi-input: build a per-ticker slice and apply _atr_naive to (high, low, close).
    parts: list[pd.Series] = []
    for _, g in prices.groupby("ticker", sort=False):
        atr_naive = _atr_naive(g["high"], g["low"], g["close"], period)
        shifted = atr_naive.shift(_PAST_SHIFT)
        shifted.index = g.index
        parts.append(shifted)
    if parts:
        col_vals = pd.concat(parts).reindex(prices.index)
    else:
        col_vals = pd.Series(np.nan, index=prices.index)
    return prices.assign(**{col: col_vals})


def add_bollinger(prices: pd.DataFrame, *, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Append ``bb_mid``, ``bb_upper``, ``bb_lower`` (past-only)."""
    out = prices.copy()
    naive = _per_ticker_multi_col(out, lambda g: _bollinger_naive(g["close"], window, num_std))
    for col in ("bb_mid", "bb_upper", "bb_lower"):
        out[col] = naive[col]
    return out


# ---------------------------------------------------------------------------
# Config + orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for the technical indicators block.

    ``indicators`` is a tuple of indicator names recognized by
    :func:`add_technical_features`. ``describe()`` returns the list of output
    column names the block emits (used by tests and the lookahead guard to know
    which columns to recompute).
    """

    indicators: tuple[str, ...] = (
        "returns",
        "sma",
        "ema",
        "rsi",
        "macd",
        "volatility",
        "volume_zscore",
        "atr",
        "bollinger",
    )
    return_horizons: tuple[int, ...] = (1, 5, 10, 21)
    sma_windows: tuple[int, ...] = (5, 10, 21, 63)
    ema_windows: tuple[int, ...] = (5, 10, 21, 63)
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volatility_windows: tuple[int, ...] = (21, 63)
    volume_zscore_window: int = 21
    atr_period: int = 14
    bollinger_window: int = 20
    bollinger_num_std: float = 2.0

    def describe(self) -> list[str]:
        """Return the list of output column names emitted by this config."""
        cols: list[str] = []
        if "returns" in self.indicators:
            cols += [f"ret_{n}d_log" for n in self.return_horizons]
        if "sma" in self.indicators:
            cols += [f"sma_{w}" for w in self.sma_windows]
        if "ema" in self.indicators:
            cols += [f"ema_{w}" for w in self.ema_windows]
        if "rsi" in self.indicators:
            cols.append(f"rsi_{self.rsi_period}")
        if "macd" in self.indicators:
            cols += ["macd", "macd_signal", "macd_hist"]
        if "volatility" in self.indicators:
            cols += [f"volatility_{w}" for w in self.volatility_windows]
        if "volume_zscore" in self.indicators:
            cols.append(f"volume_zscore_{self.volume_zscore_window}")
        if "atr" in self.indicators:
            cols.append(f"atr_{self.atr_period}")
        if "bollinger" in self.indicators:
            cols += ["bb_mid", "bb_upper", "bb_lower"]
        return cols


DEFAULT_INDICATORS = FeatureConfig()
TECHNICAL_FEATURE_COLUMNS: list[str] = DEFAULT_INDICATORS.describe()


def add_technical_features(
    prices: pd.DataFrame, *, indicators: FeatureConfig | None = None
) -> pd.DataFrame:
    """Append all configured technical indicators to ``prices`` (past-only).

    Parameters
    ----------
    prices:
        Long-form DataFrame with the canonical S1 price columns (see
        :data:`equity.data.schema.CANONICAL_PRICE_COLUMNS`). Sorted ascending by
        ``period_close_ts`` within ticker (this function sorts defensively).
    indicators:
        Optional :class:`FeatureConfig`. Defaults to :data:`DEFAULT_INDICATORS`.

    Returns
    -------
    pd.DataFrame
        ``prices`` with indicator columns appended. Does NOT mutate the input.
    """
    cfg = indicators or DEFAULT_INDICATORS
    out = prices.copy()
    if out.empty:
        return out
    # Defensive per-ticker sort: every rolling op below relies on ascending
    # time order within each ticker group.
    out = out.sort_values(["ticker", "period_close_ts"]).reset_index(drop=True)
    if "returns" in cfg.indicators:
        out = add_returns(out, horizons=cfg.return_horizons)
    if "sma" in cfg.indicators:
        out = add_sma(out, windows=cfg.sma_windows)
    if "ema" in cfg.indicators:
        out = add_ema(out, windows=cfg.ema_windows)
    if "rsi" in cfg.indicators:
        out = add_rsi(out, period=cfg.rsi_period)
    if "macd" in cfg.indicators:
        out = add_macd(out, fast=cfg.macd_fast, slow=cfg.macd_slow, signal=cfg.macd_signal)
    if "volatility" in cfg.indicators:
        out = add_volatility(out, windows=cfg.volatility_windows)
    if "volume_zscore" in cfg.indicators:
        out = add_volume_zscore(out, window=cfg.volume_zscore_window)
    if "atr" in cfg.indicators:
        out = add_atr(out, period=cfg.atr_period)
    if "bollinger" in cfg.indicators:
        out = add_bollinger(out, window=cfg.bollinger_window, num_std=cfg.bollinger_num_std)
    return out


# ---------------------------------------------------------------------------
# NAIVE indicator specs -- the generic recompute core (D3).
#
# Maps each past-only feature column to a per-ticker single-ticker spec fn that
# returns the NAIVE (current-row-inclusive) indicator series. The lookahead
# guard feeds each ticker's ``prices[:t]`` slice through the matching fn and
# compares the LAST value to the stored feature at row t -- this is the exact
# leak the guard catches (FOOTGUN #1). New indicators (and sentiment roll
# features, later) are added by extending this dict -- no guard rewrite needed.
#
# Each fn takes a SINGLE-TICKER prices frame (sorted ascending by
# period_close_ts) and returns a Series aligned to that frame's index.
# ---------------------------------------------------------------------------


def _build_naive_specs(cfg: FeatureConfig) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    specs: dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
    if "returns" in cfg.indicators:
        for n in cfg.return_horizons:
            specs[f"ret_{n}d_log"] = (lambda n: lambda p: _ret_n_naive(p["close"], n))(n)
    if "sma" in cfg.indicators:
        for w in cfg.sma_windows:
            specs[f"sma_{w}"] = (lambda w: lambda p: _sma_naive(p["close"], w))(w)
    if "ema" in cfg.indicators:
        for w in cfg.ema_windows:
            specs[f"ema_{w}"] = (lambda w: lambda p: _ema_naive(p["close"], w))(w)
    if "rsi" in cfg.indicators:
        specs[f"rsi_{cfg.rsi_period}"] = lambda p: _rsi_naive(p["close"], cfg.rsi_period)
    if "macd" in cfg.indicators:
        # MACD spec returns a Series for the "macd" line; the guard also needs
        # macd_signal and macd_hist. We register each via a fn that returns the
        # corresponding column of the naive MACD frame.
        def _macd_col(name: str):
            return lambda p: _macd_naive(p["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)[
                name
            ]

        specs["macd"] = _macd_col("macd")
        specs["macd_signal"] = _macd_col("macd_signal")
        specs["macd_hist"] = _macd_col("macd_hist")
    if "volatility" in cfg.indicators:
        for w in cfg.volatility_windows:
            specs[f"volatility_{w}"] = (lambda w: lambda p: _volatility_naive(p["close"], w))(w)
    if "volume_zscore" in cfg.indicators:
        specs[f"volume_zscore_{cfg.volume_zscore_window}"] = lambda p: _volume_zscore_naive(
            p["volume"], cfg.volume_zscore_window
        )
    if "atr" in cfg.indicators:
        specs[f"atr_{cfg.atr_period}"] = lambda p: _atr_naive(
            p["high"], p["low"], p["close"], cfg.atr_period
        )
    if "bollinger" in cfg.indicators:
        for col, name in (
            ("bb_mid", "bb_mid"),
            ("bb_upper", "bb_upper"),
            ("bb_lower", "bb_lower"),
        ):
            specs[name] = (
                lambda name: (
                    lambda p: _bollinger_naive(
                        p["close"], cfg.bollinger_window, cfg.bollinger_num_std
                    )[name]
                )
            )(name)
    return specs


NAIVE_INDICATOR_SPECS: dict[str, Callable[[pd.DataFrame], pd.Series]] = _build_naive_specs(
    DEFAULT_INDICATORS
)


__all__ = [
    "FeatureConfig",
    "DEFAULT_INDICATORS",
    "TECHNICAL_FEATURE_COLUMNS",
    "NAIVE_INDICATOR_SPECS",
    "add_technical_features",
    "add_returns",
    "add_sma",
    "add_ema",
    "add_rsi",
    "add_macd",
    "add_volatility",
    "add_volume_zscore",
    "add_atr",
    "add_bollinger",
]
