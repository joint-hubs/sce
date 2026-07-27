"""
@module: tests.equity.test_features_build
@depends: equity.features.build, equity.features.technical, equity.features.lag
@exports:
@data_flow: prices (+ optional sentiment_per_period) -> build_features ->
            flat past-only feature matrix; e2e with monkeypatched PROJECT_ROOT
            for any parquet write (mirrors test_sentiment_e2e.py).

S3.4 build orchestration tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity.features.build import build_features
from equity.features.lag import DEFAULT_LAG_WINDOWS
from equity.features.technical import DEFAULT_INDICATORS


def _prices_fixture(n: int = 30, n_tickers: int = 2) -> pd.DataFrame:
    # 16:00 ET (the canonical session close per equity/data/schema.py).
    ts = pd.date_range("2024-01-01 16:00", periods=n, freq="D", tz="America/New_York")
    frames = []
    for t in range(n_tickers):
        rng = np.random.default_rng(10 + t)
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": f"TK{t}",
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
        )
    return pd.concat(frames, ignore_index=True)


def _sentiment_fixture(n: int = 30, n_tickers: int = 2) -> pd.DataFrame:
    """Mirrors FOC-49 aggregate_per_period output: UTC period_close_ts.

    Times are aligned to the prices fixture's period close (16:00 ET = 20:00 UTC
    for midnight-ET America/New_York dates -- but prices uses midnight NY
    which is 05:00 UTC; we use 20:00 UTC to match the canonical 16:00 ET close).
    """
    # 16:00 ET = 21:00 UTC in January (EST, UTC-5); the prices fixture uses
    # 2024-01-01 16:00 America/New_York, so align sentiment to 21:00 UTC.
    ts = pd.date_range("2024-01-01 21:00", periods=n, freq="D", tz="UTC")
    rows = []
    for t in range(n_tickers):
        rng = np.random.default_rng(100 + t)
        for i in range(n):
            rows.append(
                {
                    "ticker": f"TK{t}",
                    "period_close_ts": ts[i],
                    "sentiment_score": float(rng.uniform(-1, 1)),
                    "sentiment_pos": float(rng.uniform(0, 1)),
                    "sentiment_neg": float(rng.uniform(0, 1)),
                    "sentiment_neu": float(rng.uniform(0, 1)),
                    "n_articles": int(rng.integers(0, 5)),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Shape + column contract
# ---------------------------------------------------------------------------


def test_build_features_no_sentiment_shape_and_columns():
    prices = _prices_fixture(n=30, n_tickers=2)
    feats = build_features(prices)
    # One row per (ticker, period_close_ts).
    assert len(feats) == len(prices)
    # Technical columns present.
    tech_cols = DEFAULT_INDICATORS.describe()
    for c in tech_cols:
        assert c in feats.columns
    # No sentiment columns (sentiment_per_period=None).
    assert "sentiment_score" not in feats.columns
    # Lag columns over technical base cols only.
    for c in tech_cols:
        for w in DEFAULT_LAG_WINDOWS:
            assert f"{c}_lag{w}" in feats.columns
            assert f"{c}_rollmean{w}" in feats.columns
            assert f"{c}_rollstd{w}" in feats.columns


def test_build_features_with_sentiment_left_join():
    prices = _prices_fixture(n=30, n_tickers=2)
    sent = _sentiment_fixture(n=30, n_tickers=2)
    feats = build_features(prices, sent)
    # Sentiment base cols present (LEFT-JOINed, filled).
    for c in ("sentiment_score", "sentiment_pos", "sentiment_neg", "sentiment_neu", "n_articles"):
        assert c in feats.columns
        # No NaN in sentiment cols (filled to 0 per D4).
        assert not feats[c].isna().any()
    # Lag columns over sentiment base cols present.
    for c in ("sentiment_score", "sentiment_pos", "sentiment_neg", "sentiment_neu", "n_articles"):
        for w in DEFAULT_LAG_WINDOWS:
            assert f"{c}_lag{w}" in feats.columns


def test_build_features_missing_sentiment_periods_filled_zero():
    """A (ticker, period) missing from sentiment_per_period -> NaN -> 0 (D4)."""
    prices = _prices_fixture(n=10, n_tickers=1)
    # Sentiment covers only the first 3 periods of TK0.
    sent = _sentiment_fixture(n=3, n_tickers=1)
    feats = build_features(prices, sent)
    # sentiment_score for the first 3 rows of TK0 should be non-zero (filled
    # from sent); rows 3..9 should be 0.0 (filled from NaN).
    tk = feats[feats["ticker"] == "TK0"].sort_values("period_close_ts").reset_index(drop=True)
    assert (tk["sentiment_score"].iloc[:3] != 0).any()
    assert (tk["sentiment_score"].iloc[3:] == 0.0).all()
    assert (tk["n_articles"].iloc[3:] == 0.0).all()


def test_build_features_empty_sentiment_frame():
    """Empty sentiment frame -> zeroed sentiment columns (downstream stable)."""
    prices = _prices_fixture(n=10, n_tickers=1)
    sent = pd.DataFrame(
        columns=[
            "ticker",
            "period_close_ts",
            "sentiment_score",
            "sentiment_pos",
            "sentiment_neg",
            "sentiment_neu",
            "n_articles",
        ]
    )
    feats = build_features(prices, sent)
    for c in ("sentiment_score", "sentiment_pos", "sentiment_neg", "sentiment_neu", "n_articles"):
        assert c in feats.columns
        assert (feats[c] == 0.0).all()


def test_build_features_tz_canonicalization():
    """Prices America/New_York + sentiment UTC -> output period_close_ts in UTC."""
    prices = _prices_fixture(n=10, n_tickers=1)
    sent = _sentiment_fixture(n=10, n_tickers=1)
    feats = build_features(prices, sent)
    # The output period_close_ts is the prices' (now UTC-converted).
    assert feats["period_close_ts"].dt.tz is not None
    # The merge succeeded -- sentiment_score is non-NaN everywhere.
    assert not feats["sentiment_score"].isna().any()


def test_build_features_past_only_invariant():
    """The lag layer must be past-only: ``x_lag1[t] == x[t-1]`` for feature cols."""
    prices = _prices_fixture(n=15, n_tickers=1)
    feats = build_features(prices)
    s = feats.sort_values(["ticker", "period_close_ts"]).reset_index(drop=True)
    for i in range(1, len(s)):
        if pd.isna(s["sma_5"].iloc[i - 1]):
            assert pd.isna(s["sma_5_lag1"].iloc[i])
        else:
            assert s["sma_5_lag1"].iloc[i] == pytest.approx(s["sma_5"].iloc[i - 1], abs=1e-9)
        if pd.isna(s["rsi_14"].iloc[i - 1]):
            assert pd.isna(s["rsi_14_lag1"].iloc[i])
        else:
            assert s["rsi_14_lag1"].iloc[i] == pytest.approx(s["rsi_14"].iloc[i - 1], abs=1e-9)


def test_build_features_per_ticker_independence():
    """Lag windows must NOT bleed across ticker boundary (D7)."""
    prices = _prices_fixture(n=20, n_tickers=2)
    feats = build_features(prices)
    s = feats.sort_values(["ticker", "period_close_ts"]).reset_index(drop=True)
    # First row of each ticker's sma_5_lag1 must be NaN (no prior in that ticker).
    for ticker in ("TK0", "TK1"):
        first = s[s["ticker"] == ticker].iloc[0]
        assert pd.isna(first["sma_5_lag1"])
        assert pd.isna(first["rsi_14_lag1"])


def test_build_features_does_not_mutate_input():
    prices = _prices_fixture(n=20, n_tickers=1)
    original_cols = prices.columns.tolist()
    _ = build_features(prices)
    assert prices.columns.tolist() == original_cols


def test_build_features_lag_base_cols_subset():
    """``lag_base_cols`` lets the caller lag only a subset (e.g. only sentiment)."""
    prices = _prices_fixture(n=15, n_tickers=1)
    sent = _sentiment_fixture(n=15, n_tickers=1)
    feats = build_features(prices, sent, lag_base_cols=["sentiment_score"])
    # Only sentiment_score lags are emitted -- technical cols have no lag cols.
    assert "sentiment_score_lag1" in feats.columns
    assert "sma_5_lag1" not in feats.columns


def test_build_features_lookahead_guard_clean():
    """The full build_features output must PASS the lookahead guard."""
    from equity.diagnostics.lookahead_indicator import run_lookahead_indicator

    prices = _prices_fixture(n=40, n_tickers=2)
    feats = build_features(prices)
    result = run_lookahead_indicator(feats, prices)
    assert result["pass"] is True
    assert result["n_violations"] == 0


# ---------------------------------------------------------------------------
# Round-1 review fixes: BLOCKER 3 (merge key uniqueness / row count), TEST 9
# (tz-naive canonicalization), SUBSTANTIVE 15 (has_sentiment flag).
# ---------------------------------------------------------------------------


def test_build_features_rejects_duplicate_sentiment_keys():
    """BLOCKER 3: duplicate (ticker, period_close_ts) keys in the sentiment
    frame would fan-out the LEFT-JOIN -> ValueError."""
    prices = _prices_fixture(n=10, n_tickers=1)
    sent = _sentiment_fixture(n=10, n_tickers=1)
    # Duplicate the first row -> duplicate (ticker, period_close_ts) key.
    sent = pd.concat([sent, sent.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate.*ticker.*period_close_ts"):
        build_features(prices, sent)


def test_build_features_row_count_preserved():
    """BLOCKER 3: post-merge row count must equal input prices row count."""
    prices = _prices_fixture(n=15, n_tickers=2)
    sent = _sentiment_fixture(n=15, n_tickers=2)
    feats = build_features(prices, sent)
    assert len(feats) == len(prices)


def test_canonicalize_tz_naive_sentiment():
    """TEST 9: the tz-naive -> tz_localize('UTC') branch of
    :func:`_canonicalize_tz_utc` (currently only the tz-aware branch is
    covered by the other tests)."""
    from equity.features.build import _canonicalize_tz_utc

    df = pd.DataFrame(
        {
            "period_close_ts": pd.date_range("2024-01-01", periods=5, freq="D"),  # tz-naive
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    out = _canonicalize_tz_utc(df, "period_close_ts")
    assert out["period_close_ts"].dt.tz is not None
    assert str(out["period_close_ts"].dt.tz) == "UTC"
    # Values unchanged.
    assert (out["x"] == df["x"]).all()


def test_build_features_has_sentiment_flag():
    """SUBSTANTIVE 15: ``has_sentiment`` bool flag disambiguates the NaN->0
    fill. True where the LEFT-JOIN matched a non-null sentiment row
    (n_articles > 0 after fill); False where the join missed (no articles)."""
    prices = _prices_fixture(n=10, n_tickers=1)
    # Sentiment covers only the first 3 periods; the rest miss the join.
    sent = _sentiment_fixture(n=3, n_tickers=1)
    feats = build_features(prices, sent)
    assert "has_sentiment" in feats.columns
    tk = feats[feats["ticker"] == "TK0"].sort_values("period_close_ts").reset_index(drop=True)
    # The sentiment fixture may produce n_articles=0 for some rows even when
    # the join matched; the flag is True only where n_articles > 0 after fill.
    # Verify the flag is a bool and tracks n_articles > 0 row-by-row.
    assert tk["has_sentiment"].dtype == bool
    expected_flag = tk["n_articles"] > 0
    assert (tk["has_sentiment"] == expected_flag).all()


def test_build_features_has_sentiment_flag_empty_frame():
    """SUBSTANTIVE 15: an empty sentiment frame -> has_sentiment=False for all
    rows (no articles anywhere)."""
    prices = _prices_fixture(n=8, n_tickers=1)
    sent = pd.DataFrame(
        columns=[
            "ticker",
            "period_close_ts",
            "sentiment_score",
            "sentiment_pos",
            "sentiment_neg",
            "sentiment_neu",
            "n_articles",
        ]
    )
    feats = build_features(prices, sent)
    assert "has_sentiment" in feats.columns
    assert (feats["has_sentiment"] == False).all()  # noqa: E712


def test_build_features_no_sentiment_has_no_flag():
    """SUBSTANTIVE 15: when sentiment_per_period is None, the has_sentiment
    column is NOT added (no sentiment block ran)."""
    prices = _prices_fixture(n=8, n_tickers=1)
    feats = build_features(prices)
    assert "has_sentiment" not in feats.columns
