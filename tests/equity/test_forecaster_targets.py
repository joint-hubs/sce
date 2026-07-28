"""
@module: tests.equity.test_forecaster_targets
@depends: equity.forecaster.targets
@exports:
@data_flow: synthetic prices -> add_forward_targets -> ret_hN columns

S5.1 target-builder unit tests. Structural + numeric on a hand-checked path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity.forecaster.targets import (
    add_forward_targets,
    forward_target_col,
    list_forward_target_cols,
)


def _mini_prices(n: int = 10, n_tickers: int = 2) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01 16:00", periods=n, freq="D", tz="America/New_York")
    frames = []
    for t in range(n_tickers):
        # Deterministic geometric path so we can hand-check log returns.
        close = 100.0 * (1.01 ** np.arange(n)) * (1.0 + 0.1 * t)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": f"TK{t}",
                    "period_close_ts": ts,
                    "close": close,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def test_forward_target_col_name() -> None:
    assert forward_target_col(1) == "ret_h1"
    assert forward_target_col(63) == "ret_h63"
    with pytest.raises(ValueError):
        forward_target_col(0)


def test_list_forward_target_cols_default() -> None:
    cols = list_forward_target_cols()
    assert cols == ("ret_h1", "ret_h5", "ret_h10", "ret_h21", "ret_h63")


def test_add_forward_targets_columns_and_tail_nan() -> None:
    prices = _mini_prices(n=10, n_tickers=2)
    out = add_forward_targets(prices, horizons=(1, 5))
    assert "ret_h1" in out.columns
    assert "ret_h5" in out.columns
    # Per ticker, last N rows are NaN for horizon N.
    for t, g in out.groupby("ticker"):
        assert g["ret_h1"].iloc[-1] != g["ret_h1"].iloc[-1]  # NaN
        assert g["ret_h1"].iloc[:-1].notna().all()
        assert g["ret_h5"].iloc[-5:].isna().all()
        assert g["ret_h5"].iloc[:-5].notna().all()


def test_add_forward_targets_numeric_formula() -> None:
    prices = _mini_prices(n=8, n_tickers=1)
    out = add_forward_targets(prices, horizons=(1, 3))
    g = out.sort_values("period_close_ts")
    close = g["close"].to_numpy()
    expected_h1 = np.log(close[1:] / close[:-1])
    got_h1 = g["ret_h1"].iloc[:-1].to_numpy()
    np.testing.assert_allclose(got_h1, expected_h1, rtol=0, atol=1e-12)

    expected_h3 = np.log(close[3:] / close[:-3])
    got_h3 = g["ret_h3"].iloc[:-3].to_numpy()
    np.testing.assert_allclose(got_h3, expected_h3, rtol=0, atol=1e-12)


def test_add_forward_targets_no_cross_ticker_bleed() -> None:
    """Last rows of TK0 must not pick up TK1's future closes."""
    prices = _mini_prices(n=5, n_tickers=2)
    out = add_forward_targets(prices, horizons=(2,))
    tk0 = out.loc[out["ticker"] == "TK0"].sort_values("period_close_ts")
    # Last 2 of TK0 are NaN (no future inside the ticker group).
    assert tk0["ret_h2"].iloc[-2:].isna().all()
    # Third-from-last uses TK0's own close[t+2].
    c = tk0["close"].to_numpy()
    expected = np.log(c[2] / c[0])
    assert np.isclose(tk0["ret_h2"].iloc[0], expected)


def test_add_forward_targets_rejects_empty_horizons() -> None:
    prices = _mini_prices(n=5, n_tickers=1)
    with pytest.raises(ValueError):
        add_forward_targets(prices, horizons=())
