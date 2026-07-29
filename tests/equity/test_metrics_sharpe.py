"""
@module: tests.equity.test_metrics_sharpe
@depends: equity.metrics.sharpe
@exports:
@data_flow: hand fixtures -> decile L/S / Sharpe / Sortino / select_horizon

S6.3 portfolio metric unit tests. Exact values on small hand-computed fixtures.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from equity.metrics.sharpe import (
    aggregate_portfolio,
    decile_long_short_returns,
    select_horizon,
    sharpe_ratio,
    sortino_ratio,
)


def test_sharpe_ratio_hand_computed() -> None:
    # r = [0.01, -0.005, 0.02, 0.0]  (sample)
    r = np.array([0.01, -0.005, 0.02, 0.0])
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    expected = mean / std * math.sqrt(252)
    assert sharpe_ratio(r, periods_per_year=252) == pytest.approx(expected)


def test_sharpe_ratio_zero_std_nan() -> None:
    assert math.isnan(sharpe_ratio([0.1, 0.1, 0.1]))
    assert math.isnan(sharpe_ratio([]))
    assert math.isnan(sharpe_ratio([np.nan, np.nan]))


def test_sortino_ratio_hand_computed() -> None:
    # All-positive -> no downside -> NaN
    assert math.isnan(sortino_ratio([0.01, 0.02, 0.0]))
    # Mixed: downside = negatives only
    r = np.array([0.02, -0.01, 0.01, -0.03, 0.005])
    mean = float(np.mean(r))
    downside = r[r < 0.0]
    dstd = float(np.std(downside, ddof=1))
    expected = mean / dstd * math.sqrt(252)
    assert sortino_ratio(r, periods_per_year=252) == pytest.approx(expected)


def test_decile_long_short_known_ranking() -> None:
    """10 instruments at one timestamp: pred ranks 0..9, ret = pred.

    With 10 equal bins the top bin is the single highest-pred row (ret=0.09)
    and the bottom is the lowest (ret=0.00) → L/S = 0.09.
    """
    ts = pd.Timestamp("2024-01-02 16:00", tz="UTC")
    preds = np.arange(10, dtype=float) * 0.01  # 0.00 .. 0.09
    rets = preds.copy()
    df = pd.DataFrame(
        {
            "ticker": [f"T{i}" for i in range(10)],
            "period_close_ts": [ts] * 10,
            "pred_h1": preds,
            "ret_h1": rets,
        }
    )
    ls = decile_long_short_returns(
        df, ret_col="ret_h1", pred_col="pred_h1", time_col="period_close_ts"
    )
    assert len(ls) == 1
    assert float(ls.iloc[0]) == pytest.approx(0.09)


def test_decile_long_short_two_timestamps() -> None:
    """Two timestamps, 4 names each: n_bins=4, L/S = top−bottom single names."""
    rows = []
    for day, offset in [(1, 0.0), (2, 1.0)]:
        ts = pd.Timestamp(f"2024-01-0{day} 16:00", tz="UTC")
        for i, (p, r) in enumerate([(0.0, 0.1), (1.0, 0.2), (2.0, 0.3), (3.0, 0.4)]):
            rows.append(
                {
                    "ticker": f"T{i}",
                    "period_close_ts": ts,
                    "pred_h5": p + offset,
                    "ret_h5": r,
                }
            )
    df = pd.DataFrame(rows)
    ls = decile_long_short_returns(
        df, ret_col="ret_h5", pred_col="pred_h5", time_col="period_close_ts"
    )
    assert len(ls) == 2
    # Top ret 0.4, bottom 0.1 → 0.3 each day
    assert float(ls.iloc[0]) == pytest.approx(0.3)
    assert float(ls.iloc[1]) == pytest.approx(0.3)


def test_decile_handles_small_cross_section() -> None:
    """n=3 < 10 → 3 bins, no crash; L/S still defined."""
    ts = pd.Timestamp("2024-06-01")
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "period_close_ts": [ts] * 3,
            "pred": [1.0, 2.0, 3.0],
            "ret": [0.1, 0.0, -0.2],
        }
    )
    ls = decile_long_short_returns(
        df, ret_col="ret", pred_col="pred", time_col="period_close_ts"
    )
    # top=C ret=-0.2, bottom=A ret=0.1 → -0.3
    assert float(ls.iloc[0]) == pytest.approx(-0.3)


def test_decile_nan_safe_skips_bad_rows() -> None:
    ts = pd.Timestamp("2024-01-01")
    df = pd.DataFrame(
        {
            "ticker": list("ABCD"),
            "period_close_ts": [ts] * 4,
            "pred": [1.0, np.nan, 3.0, 4.0],
            "ret": [0.1, 0.2, np.nan, 0.4],
        }
    )
    ls = decile_long_short_returns(
        df, ret_col="ret", pred_col="pred", time_col="period_close_ts"
    )
    # only A (pred1,ret0.1) and D (pred4,ret0.4) usable → L/S = 0.4-0.1
    assert float(ls.iloc[0]) == pytest.approx(0.3)


def test_select_horizon_picks_best_sharpe() -> None:
    """Synthetic: h=5 has clearly better long/short ordering than h=1.

    L/S portfolio return varies across days so Sharpe is defined (non-zero std).
    h=1: inverted ranking → mostly negative L/S → low/negative Sharpe.
    h=5: aligned ranking → positive L/S → high Sharpe.
    """
    rng = np.random.default_rng(0)
    ts = pd.date_range("2024-01-01", periods=20, freq="D")
    rows_h1 = []
    rows_h5 = []
    for t_idx, t in enumerate(ts):
        # Per-day signal strength varies so L/S series is non-constant.
        strength = 0.01 + 0.005 * math.sin(t_idx)
        noise = rng.normal(0.0, 0.002, size=10)
        for i in range(10):
            true_score = float(i)
            ret = strength * true_score + float(noise[i])
            rows_h1.append(
                {
                    "ticker": f"T{i}",
                    "period_close_ts": t,
                    "pred_h1": float(9 - i),  # inverted
                    "ret_h1": ret,
                    "split": "val",
                }
            )
            rows_h5.append(
                {
                    "ticker": f"T{i}",
                    "period_close_ts": t,
                    "pred_h5": float(i),  # aligned
                    "ret_h5": ret,
                    "split": "val",
                }
            )
    frames = {
        1: pd.DataFrame(rows_h1),
        5: pd.DataFrame(rows_h5),
    }
    # Sanity: h=5 sharpe strictly beats h=1 on this fixture.
    from equity.metrics.sharpe import decile_long_short_returns, sharpe_ratio

    ls1 = decile_long_short_returns(
        frames[1], ret_col="ret_h1", pred_col="pred_h1", time_col="period_close_ts"
    )
    ls5 = decile_long_short_returns(
        frames[5], ret_col="ret_h5", pred_col="pred_h5", time_col="period_close_ts"
    )
    assert sharpe_ratio(ls5) > sharpe_ratio(ls1)

    chosen = select_horizon(frames, horizons=(1, 5), criterion="sharpe")
    assert chosen == 5

    # Deterministic across calls
    assert select_horizon(frames, horizons=(1, 5), criterion="sharpe") == 5


def test_select_horizon_tie_breaks_smallest() -> None:
    """Identical frames → identical scores → smallest h wins."""
    rng = np.random.default_rng(1)
    ts = pd.date_range("2024-01-01", periods=12, freq="D")
    rows = []
    for t_idx, t in enumerate(ts):
        noise = rng.normal(0.0, 0.001, size=6)
        for i in range(6):
            ret = 0.01 * float(i) + 0.002 * math.sin(t_idx) + float(noise[i])
            rows.append(
                {
                    "ticker": f"T{i}",
                    "period_close_ts": t,
                    "pred_h1": float(i),
                    "ret_h1": ret,
                    "pred_h5": float(i),
                    "ret_h5": ret,
                    "split": "val",
                }
            )
    df = pd.DataFrame(rows)
    assert select_horizon(df, horizons=(5, 1), criterion="sharpe") == 1


def test_aggregate_portfolio_schema() -> None:
    rng = np.random.default_rng(2)
    ts = pd.date_range("2024-01-01", periods=15, freq="D")
    rows = []
    for t_idx, t in enumerate(ts):
        noise = rng.normal(0.0, 0.002, size=8)
        for i in range(8):
            rows.append(
                {
                    "ticker": f"T{i}",
                    "period_close_ts": t,
                    "pred_h1": float(i),
                    "ret_h1": 0.01 * float(i) + 0.003 * math.cos(t_idx) + float(noise[i]),
                    "split": "val",
                }
            )
    fold0 = {1: pd.DataFrame(rows)}
    out = aggregate_portfolio([fold0], horizons=(1,), criterion="sharpe")
    assert out["chosen_horizon"] == 1
    assert 1 in out["per_horizon"]
    assert "sharpe" in out["per_horizon"][1]
    assert "sortino" in out["per_horizon"][1]
    assert np.isfinite(out["per_horizon"][1]["sharpe"])