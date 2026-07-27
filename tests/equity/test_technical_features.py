"""
@module: tests.equity.test_technical_features
@depends: equity.features.technical
@exports:
@data_flow: hand-built prices fixture -> add_technical_features -> numeric
            values asserted via pytest.approx (SMA/EMA/RSI abs=1e-9,
            MACD abs=1e-6)

S3.1 tests. Hand-built DataFrames with hand-checked numeric values, per the
repo test conventions (no shared conftest fixtures for equity tests).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from equity.features.technical import (
    DEFAULT_INDICATORS,
    NAIVE_INDICATOR_SPECS,
    TECHNICAL_FEATURE_COLUMNS,
    FeatureConfig,
    add_technical_features,
)


def _prices_fixture(n: int = 30, seed: int = 42) -> pd.DataFrame:
    """Single-ticker AAPL prices, ascending by period_close_ts, tz-aware
    America/New_York (the S1 canonical schema)."""
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="America/New_York")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "ticker": "AAPL",
            "period_close_ts": ts,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "adj_close": close,
            "volume": rng.integers(1000, 10000, n).astype(float),
            "hlc_average": (close * 1.01 + close * 0.98 + close) / 3.0,
        }
    )


def _two_ticker_fixture(n: int = 30) -> pd.DataFrame:
    """Two tickers (AAPL, MSFT) interleaved in time -- tests per-ticker
    independence (D7)."""
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="America/New_York")
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(22)
    a = pd.DataFrame(
        {
            "ticker": "AAPL",
            "period_close_ts": ts,
            "open": 1.0,
            "high": 1.5,
            "low": 0.5,
            "close": np.cumprod(1 + rng_a.normal(0, 0.01, n)),
            "adj_close": 1.0,
            "volume": 1000.0,
            "hlc_average": 1.0,
        }
    )
    b = pd.DataFrame(
        {
            "ticker": "MSFT",
            "period_close_ts": ts,
            "open": 1.0,
            "high": 1.5,
            "low": 0.5,
            "close": np.cumprod(1 + rng_b.normal(0, 0.01, n)),
            "adj_close": 1.0,
            "volume": 1000.0,
            "hlc_average": 1.0,
        }
    )
    # Interleave so any cross-ticker bleed would manifest as a wrong value.
    return (
        pd.concat([a, b], ignore_index=True).sample(frac=1, random_state=99).reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# SMA: hand-checked
# ---------------------------------------------------------------------------


def test_sma_hand_checked_values():
    """sma_5[t] = mean(close[t-5 .. t-1]) -- past-only (excludes close[t])."""
    df = _prices_fixture(n=10, seed=1)
    # Force a known close trajectory: 100, 101, 102, 103, 104, 105, 106, 107, 108, 109
    df["close"] = 100.0 + np.arange(10)
    feats = add_technical_features(df)
    # sma_5 at row index 5 (close=105): mean(close[0..4]) = mean(100,101,102,103,104) = 102.0
    assert feats["sma_5"].iloc[5] == pytest.approx(102.0, abs=1e-9)
    # sma_5 at row 6: mean(close[1..5]) = mean(101,102,103,104,105) = 103.0
    assert feats["sma_5"].iloc[6] == pytest.approx(103.0, abs=1e-9)
    # sma_5 at row 9: mean(close[4..8]) = mean(104,105,106,107,108) = 106.0
    assert feats["sma_5"].iloc[9] == pytest.approx(106.0, abs=1e-9)
    # Warmup: first 5 rows are NaN (D5)
    assert feats["sma_5"].iloc[:5].isna().all()


def test_sma_excludes_current_row_past_only():
    """Critical PIT check: sma_5[t] must NOT include close[t]."""
    df = _prices_fixture(n=10, seed=2)
    df["close"] = 100.0 + np.arange(10)
    feats = add_technical_features(df)
    # If sma_5[t] had included close[t] (the leak), at row 5 it would be
    # mean(close[1..5]) = mean(101,102,103,104,105) = 103.0. Past-only value is
    # mean(close[0..4]) = 102.0.
    assert feats["sma_5"].iloc[5] == pytest.approx(102.0, abs=1e-9)
    assert feats["sma_5"].iloc[5] != pytest.approx(103.0, abs=1e-9)


# ---------------------------------------------------------------------------
# EMA: hand-checked
# ---------------------------------------------------------------------------


def test_ema_hand_checked_values():
    """ema_5[t] = ewm(close, span=5, adjust=False).shift(1) -- past-only."""
    df = _prices_fixture(n=10, seed=3)
    df["close"] = 100.0 + np.arange(10)
    feats = add_technical_features(df)
    # Compute expected: naive ema (current-row-inclusive), shifted by 1.
    expected_naive = df["close"].ewm(span=5, adjust=False).mean()
    expected_past_only = expected_naive.shift(1)
    for i in range(1, 10):
        assert feats["ema_5"].iloc[i] == pytest.approx(expected_past_only.iloc[i], abs=1e-9)


# ---------------------------------------------------------------------------
# RSI: hand-checked
# ---------------------------------------------------------------------------


def test_rsi_hand_checked_values():
    """rsi_14[t] = Wilder RSI(14).shift(1) -- past-only.

    Wilder's ewm(alpha=1/period) is unconditionally defined from the first
    non-NaN delta (row 1) onward -- there's no min_periods warmup. The past-only
    shift pushes the first valid value from row 2 to row 3.
    """
    df = _prices_fixture(n=30, seed=4)
    df["close"] = 100.0 + np.arange(30)  # monotonically increasing
    feats = add_technical_features(df)
    rsi = feats["rsi_14"]
    # Rows 0, 1, 2 are NaN (row 0: no prior delta; row 1: naive valid but
    # shifted out; row 2: shifted value of row 1 which exists -- actually row 2
    # is the first shifted value). Verify the warmup/shift boundary.
    assert pd.isna(rsi.iloc[0])
    assert pd.isna(rsi.iloc[1])
    # Row 2 onward: all up moves -> RSI = 100.
    for i in range(2, len(rsi)):
        assert rsi.iloc[i] == pytest.approx(100.0, abs=1e-9)


def test_rsi_all_down_is_zero():
    """Monotonically decreasing close -> RSI = 0 once warmed up."""
    df = _prices_fixture(n=30, seed=5)
    df["close"] = 200.0 - np.arange(30)
    feats = add_technical_features(df)
    rsi = feats["rsi_14"]
    assert pd.isna(rsi.iloc[0])
    assert pd.isna(rsi.iloc[1])
    for i in range(2, len(rsi)):
        assert rsi.iloc[i] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# MACD: hand-checked (cumulative drift tolerance)
# ---------------------------------------------------------------------------


def test_macd_hand_checked_values():
    """macd[t] = (ema_fast - ema_slow).shift(1); macd_signal = ema(macd, 9).shift(1)."""
    df = _prices_fixture(n=40, seed=6)
    feats = add_technical_features(df)
    close = df["close"]
    # Naive (current-row-inclusive) MACD.
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_naive = ema_fast - ema_slow
    signal_naive = macd_naive.ewm(span=9, adjust=False).mean()
    # Past-only: shift by 1.
    expected_macd = macd_naive.shift(1)
    expected_signal = signal_naive.shift(1)
    expected_hist = (macd_naive - signal_naive).shift(1)
    for i in range(1, len(df)):
        if not pd.isna(expected_macd.iloc[i]):
            assert feats["macd"].iloc[i] == pytest.approx(expected_macd.iloc[i], abs=1e-6)
        if not pd.isna(expected_signal.iloc[i]):
            assert feats["macd_signal"].iloc[i] == pytest.approx(expected_signal.iloc[i], abs=1e-6)
        if not pd.isna(expected_hist.iloc[i]):
            assert feats["macd_hist"].iloc[i] == pytest.approx(expected_hist.iloc[i], abs=1e-6)


# ---------------------------------------------------------------------------
# Per-ticker independence (D7)
# ---------------------------------------------------------------------------


def test_per_ticker_independence_sma():
    """A rolling window must NEVER bleed across the ticker boundary (D7)."""
    df = _two_ticker_fixture(n=30)
    feats = add_technical_features(df)
    # Sort for deterministic comparison.
    s = feats.sort_values(["ticker", "period_close_ts"]).reset_index(drop=True)
    # Recompute per-ticker sma_5 in pure pandas and compare.
    for ticker in ("AAPL", "MSFT"):
        g = s[s["ticker"] == ticker].reset_index(drop=True)
        expected = g["close"].rolling(5).mean().shift(1)
        for i in range(len(g)):
            if pd.isna(expected.iloc[i]):
                assert pd.isna(g["sma_5"].iloc[i])
            else:
                assert g["sma_5"].iloc[i] == pytest.approx(expected.iloc[i], abs=1e-9)


# ---------------------------------------------------------------------------
# Returns
# ---------------------------------------------------------------------------


def test_log_returns_hand_checked():
    """ret_1d_log[t] = log(close[t-1] / close[t-2]) -- past-only."""
    df = _prices_fixture(n=5, seed=7)
    df["close"] = [100.0, 101.0, 102.0, 103.0, 104.0]
    feats = add_technical_features(df)
    # ret_1d_log at row 2: log(close[1]/close[0]) = log(101/100)
    assert feats["ret_1d_log"].iloc[2] == pytest.approx(math.log(101 / 100), abs=1e-9)
    # ret_5d_log at row 6: log(close[5]/close[0]) -- needs n=6 rows. Use a longer frame.
    df2 = _prices_fixture(n=7, seed=7)
    df2["close"] = 100.0 + np.arange(7)
    feats2 = add_technical_features(df2)
    # ret_5d_log at row 6: log(close[5]/close[0]) = log(105/100)
    assert feats2["ret_5d_log"].iloc[6] == pytest.approx(math.log(105 / 100), abs=1e-9)


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def test_volatility_hand_checked():
    """volatility_21[t] = std(logret, 21).shift(1) -- past-only."""
    df = _prices_fixture(n=30, seed=8)
    feats = add_technical_features(df)
    logret = np.log(df["close"] / df["close"].shift(1))
    expected = logret.rolling(21, closed="right").std().shift(1)
    for i in range(len(df)):
        if pd.isna(expected.iloc[i]):
            assert pd.isna(feats["volatility_21"].iloc[i])
        else:
            assert feats["volatility_21"].iloc[i] == pytest.approx(expected.iloc[i], abs=1e-9)


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


def test_atr_hand_checked():
    """atr_14[t] = Wilder ATR(14).shift(1) -- past-only."""
    df = _prices_fixture(n=30, seed=9)
    feats = add_technical_features(df)
    # Recompute naive ATR.
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]),
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_naive = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    expected = atr_naive.shift(1)
    for i in range(len(df)):
        if pd.isna(expected.iloc[i]):
            assert pd.isna(feats["atr_14"].iloc[i])
        else:
            assert feats["atr_14"].iloc[i] == pytest.approx(expected.iloc[i], abs=1e-9)


# ---------------------------------------------------------------------------
# Bollinger
# ---------------------------------------------------------------------------


def test_bollinger_hand_checked():
    """bb_mid/upper/lower past-only (shift=1)."""
    df = _prices_fixture(n=30, seed=10)
    feats = add_technical_features(df)
    close = df["close"]
    mid_naive = close.rolling(20, closed="right").mean()
    std_naive = close.rolling(20, closed="right").std()
    expected_mid = mid_naive.shift(1)
    expected_upper = (mid_naive + 2.0 * std_naive).shift(1)
    expected_lower = (mid_naive - 2.0 * std_naive).shift(1)
    for i in range(len(df)):
        if pd.isna(expected_mid.iloc[i]):
            assert pd.isna(feats["bb_mid"].iloc[i])
        else:
            assert feats["bb_mid"].iloc[i] == pytest.approx(expected_mid.iloc[i], abs=1e-9)
            assert feats["bb_upper"].iloc[i] == pytest.approx(expected_upper.iloc[i], abs=1e-9)
            assert feats["bb_lower"].iloc[i] == pytest.approx(expected_lower.iloc[i], abs=1e-9)


# ---------------------------------------------------------------------------
# Volume z-score
# ---------------------------------------------------------------------------


def test_volume_zscore_hand_checked():
    """volume_zscore_21[t] = (volume[t-1] - mean(vol[t-21..t-1])) / std(vol[t-21..t-1]).

    Note: the naive z-score at row t uses volume[t]; past-only shifts by 1, so
    the stored value at t uses volume[t-1] and stats of vol[t-21..t-1].
    """
    df = _prices_fixture(n=30, seed=11)
    feats = add_technical_features(df)
    vol = df["volume"]
    mean_naive = vol.rolling(21, closed="right").mean()
    std_naive = vol.rolling(21, closed="right").std()
    z_naive = (vol - mean_naive) / std_naive
    expected = z_naive.shift(1)
    for i in range(len(df)):
        if pd.isna(expected.iloc[i]):
            assert pd.isna(feats["volume_zscore_21"].iloc[i])
        else:
            assert feats["volume_zscore_21"].iloc[i] == pytest.approx(expected.iloc[i], abs=1e-9)


# ---------------------------------------------------------------------------
# Config / describe
# ---------------------------------------------------------------------------


def test_feature_config_describe_columns():
    cfg = DEFAULT_INDICATORS
    cols = cfg.describe()
    # Spot-check a few column names from the PRD §5.3 spec.
    assert "ret_1d_log" in cols
    assert "sma_5" in cols
    assert "ema_21" in cols
    assert "rsi_14" in cols
    assert "macd" in cols and "macd_signal" in cols and "macd_hist" in cols
    assert "volatility_21" in cols
    assert "volume_zscore_21" in cols
    assert "atr_14" in cols
    assert "bb_mid" in cols and "bb_upper" in cols and "bb_lower" in cols
    assert TECHNICAL_FEATURE_COLUMNS == cols


def test_naive_indicator_specs_keys_match_describe():
    """D3: NAIVE_INDICATOR_SPECS covers every column in describe()."""
    cfg = DEFAULT_INDICATORS
    described = set(cfg.describe())
    speced = set(NAIVE_INDICATOR_SPECS)
    assert described == speced, (
        f"NAIVE_INDICATOR_SPECS keys diverge from describe(): "
        f"missing={described - speced}, extra={speced - described}"
    )


def test_add_technical_features_does_not_mutate_input():
    df = _prices_fixture(n=20, seed=12)
    original_cols = df.columns.tolist()
    _ = add_technical_features(df)
    assert df.columns.tolist() == original_cols


def test_add_technical_features_subset_indicators():
    cfg = FeatureConfig(indicators=("sma",))
    df = _prices_fixture(n=20, seed=13)
    feats = add_technical_features(df, indicators=cfg)
    new_cols = [c for c in feats.columns if c not in df.columns]
    # Only sma_{5,10,21,63} columns should be appended.
    assert sorted(new_cols) == sorted(["sma_5", "sma_10", "sma_21", "sma_63"])
