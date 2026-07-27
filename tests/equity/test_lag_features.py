"""
@module: tests.equity.test_lag_features
@depends: equity.features.lag
@exports:
@data_flow: hand-built feature frame -> apply_lags -> numeric checks
            (_lag1 == row t-1, _rollmean/_rollstd past-only)

S3.2 tests. Hand-built DataFrames with hand-checked numeric values, per the
repo test conventions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity.features.lag import DEFAULT_LAG_WINDOWS, LagConfig, apply_lags


def _feat_fixture(n: int = 10) -> pd.DataFrame:
    """Single-ticker frame with one base column ``x`` and a period_close_ts."""
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="America/New_York")
    return pd.DataFrame(
        {
            "ticker": "AAPL",
            "period_close_ts": ts,
            "x": np.arange(n, dtype=float),  # 0, 1, 2, ..., n-1
        }
    )


def _two_ticker_feat_fixture(n: int = 10) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="America/New_York")
    a = pd.DataFrame({"ticker": "AAPL", "period_close_ts": ts, "x": np.arange(n, dtype=float)})
    b = pd.DataFrame(
        {"ticker": "MSFT", "period_close_ts": ts, "x": 100.0 + np.arange(n, dtype=float)}
    )
    return (
        pd.concat([a, b], ignore_index=True).sample(frac=1, random_state=99).reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# _lag{N}
# ---------------------------------------------------------------------------


def test_lag1_equals_row_t_minus_1():
    """DoD S3.2: ``_lag1`` row t equals the unsourced row at t-1."""
    df = _feat_fixture(n=10)
    out = apply_lags(df, ["x"], windows=(1,))
    for i in range(1, 10):
        assert out["x_lag1"].iloc[i] == pytest.approx(out["x"].iloc[i - 1], abs=1e-9)
    # First row has no prior -> NaN.
    assert pd.isna(out["x_lag1"].iloc[0])


def test_lag3_equals_row_t_minus_3():
    df = _feat_fixture(n=10)
    out = apply_lags(df, ["x"], windows=(3,))
    for i in range(3, 10):
        assert out["x_lag3"].iloc[i] == pytest.approx(out["x"].iloc[i - 3], abs=1e-9)
    assert out["x_lag3"].iloc[:3].isna().all()


# ---------------------------------------------------------------------------
# _rollmean{N} -- past-only (closed='left')
# ---------------------------------------------------------------------------


def test_rollmean_excludes_current_row():
    """Critical PIT check: _rollmean{N}[t] must NOT include x[t]."""
    df = _feat_fixture(n=10)
    out = apply_lags(df, ["x"], windows=(3,), methods=("rolling_mean",))
    # rollmean_3[t] = mean(x[t-3..t-1]) (closed='left', excludes current).
    # At row 3: mean(x[0..2]) = mean(0,1,2) = 1.0
    assert out["x_rollmean3"].iloc[3] == pytest.approx(1.0, abs=1e-9)
    # At row 4: mean(x[1..3]) = mean(1,2,3) = 2.0
    assert out["x_rollmean3"].iloc[4] == pytest.approx(2.0, abs=1e-9)
    # If the leak had been present (closed='right'), row 3 would be mean(0,1,2,3)=1.5
    # for window=4 -- but we use window=3 here so the past-only value is mean(0,1,2)=1.0
    # and the leaked value would be mean(1,2,3)=2.0. Assert past-only.
    assert out["x_rollmean3"].iloc[3] == pytest.approx(1.0, abs=1e-9)
    assert out["x_rollmean3"].iloc[3] != pytest.approx(2.0, abs=1e-9)


def test_rollmean_hand_checked():
    df = _feat_fixture(n=10)
    out = apply_lags(df, ["x"], windows=(5,), methods=("rolling_mean",))
    # rollmean_5[t] = mean(x[t-5..t-1])
    # Row 5: mean(0,1,2,3,4) = 2.0
    assert out["x_rollmean5"].iloc[5] == pytest.approx(2.0, abs=1e-9)
    # Row 9: mean(4,5,6,7,8) = 6.0
    assert out["x_rollmean5"].iloc[9] == pytest.approx(6.0, abs=1e-9)
    # Warmup: first 5 rows NaN.
    assert out["x_rollmean5"].iloc[:5].isna().all()


def test_rollstd_hand_checked():
    """_rollstd{N}[t] = std(x[t-N..t-1]) past-only."""
    df = _feat_fixture(n=10)
    out = apply_lags(df, ["x"], windows=(5,), methods=("rolling_std",))
    # std(0,1,2,3,4) = sqrt(2.5) (sample std, ddof=1, pandas default)
    expected = float(np.std([0, 1, 2, 3, 4], ddof=1))
    assert out["x_rollstd5"].iloc[5] == pytest.approx(expected, abs=1e-9)


def test_rollmean_uses_closed_left_not_right():
    """Footgun D6: explicit closed='left' is past-only; pandas default
    closed='right' would include current row. Verify _rollmean matches
    closed='left', NOT closed='right'."""
    df = _feat_fixture(n=10)
    out = apply_lags(df, ["x"], windows=(3,), methods=("rolling_mean",))
    closed_left = df["x"].rolling(3, closed="left").mean()
    closed_right = df["x"].rolling(3, closed="right").mean()
    for i in range(len(df)):
        if pd.isna(closed_left.iloc[i]):
            assert pd.isna(out["x_rollmean3"].iloc[i])
        else:
            assert out["x_rollmean3"].iloc[i] == pytest.approx(closed_left.iloc[i], abs=1e-9)
            # And it must NOT equal closed='right' (the leak).
            assert out["x_rollmean3"].iloc[i] != pytest.approx(closed_right.iloc[i], abs=1e-9)


# ---------------------------------------------------------------------------
# Per-ticker independence (D7)
# ---------------------------------------------------------------------------


def test_apply_lags_per_ticker_independence():
    df = _two_ticker_feat_fixture(n=10)
    out = apply_lags(df, ["x"], windows=(1, 3))
    s = out.sort_values(["ticker", "period_close_ts"]).reset_index(drop=True)
    for ticker in ("AAPL", "MSFT"):
        g = s[s["ticker"] == ticker].reset_index(drop=True)
        for i in range(1, len(g)):
            assert g["x_lag1"].iloc[i] == pytest.approx(g["x"].iloc[i - 1], abs=1e-9)


def test_apply_lags_requires_ticker_column():
    df = pd.DataFrame(
        {"period_close_ts": pd.date_range("2024-01-01", periods=5), "x": [1, 2, 3, 4, 5]}
    )
    with pytest.raises(ValueError, match="ticker"):
        apply_lags(df, ["x"], windows=(1,))


def test_apply_lags_missing_base_col_raises():
    df = _feat_fixture(n=5)
    with pytest.raises(ValueError, match="not found"):
        apply_lags(df, ["nonexistent"], windows=(1,))


def test_apply_lags_does_not_mutate_input():
    df = _feat_fixture(n=5)
    original_cols = df.columns.tolist()
    _ = apply_lags(df, ["x"], windows=(1,))
    assert df.columns.tolist() == original_cols


def test_lag_config_describe():
    cfg = LagConfig(windows=(1, 3), methods=("shift", "rolling_mean"))
    cols = cfg.describe(["x", "y"])
    assert cols == [
        "x_lag1",
        "x_rollmean1",
        "x_lag3",
        "x_rollmean3",
        "y_lag1",
        "y_rollmean1",
        "y_lag3",
        "y_rollmean3",
    ]


def test_default_lag_windows():
    assert DEFAULT_LAG_WINDOWS == (1, 3, 5, 10, 21)


def test_apply_lags_default_windows():
    df = _feat_fixture(n=30)
    out = apply_lags(df, ["x"])
    # All 5 windows x 3 methods = 15 lag cols on 1 base col.
    lag_cols = [c for c in out.columns if c.startswith("x_")]
    assert len(lag_cols) == 15
    # Spot-check naming.
    for w in (1, 3, 5, 10, 21):
        assert f"x_lag{w}" in out.columns
        assert f"x_rollmean{w}" in out.columns
        assert f"x_rollstd{w}" in out.columns


# ---------------------------------------------------------------------------
# Round-1 review fixes: BLOCKER 2 (negative window rejection).
# ---------------------------------------------------------------------------


def test_apply_lags_rejects_negative_window():
    """BLOCKER 2: ``windows=(-1,)`` would call ``shift(-1)`` (future leak,
    FOOTGUN #1) -> ValueError before any transform runs."""
    df = _feat_fixture(n=10)
    with pytest.raises(ValueError, match="lag windows must be >= 1"):
        apply_lags(df, ["x"], windows=(-1,))


def test_apply_lags_rejects_zero_window():
    """BLOCKER 2 corollary: window 0 is also invalid (rolling(0) is undefined;
    shift(0) is identity -> not a lag)."""
    df = _feat_fixture(n=10)
    with pytest.raises(ValueError, match="lag windows must be >= 1"):
        apply_lags(df, ["x"], windows=(0, 3))


def test_lagconfig_rejects_negative_windows():
    """BLOCKER 2: LagConfig.__post_init__ rejects windows < 1."""
    with pytest.raises(ValueError, match="LagConfig.windows must be >= 1"):
        LagConfig(windows=(-1, 3))


def test_lagconfig_rejects_zero_windows():
    """BLOCKER 2 corollary: LagConfig rejects window 0."""
    with pytest.raises(ValueError, match="LagConfig.windows must be >= 1"):
        LagConfig(windows=(0,))


def test_lagconfig_rejects_unknown_methods():
    """BLOCKER 2 corollary: LagConfig rejects unknown method names."""
    with pytest.raises(ValueError, match="LagConfig.methods must be subset"):
        LagConfig(methods=("shift", "rolling_mean", "bogus"))


def test_lagconfig_accepts_valid_config():
    """BLOCKER 2 corollary: a valid config constructs without raising."""
    cfg = LagConfig(windows=(1, 3), methods=("shift", "rolling_mean", "rolling_std"))
    assert cfg.windows == (1, 3)
