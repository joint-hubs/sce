"""
@module: tests.equity.test_quantile
@depends: equity.forecaster.quantile
@exports:
@data_flow: panel + ret_hN -> QuantileHeadForecaster -> pred_hN_q{05,50,95}

S5.4 quantile-head tests. Asserts column presence, val-fold OOF coverage, and
a two-sided soft coverage sanity band on synthetic data. Strict [0.85, 0.95]
90% calibration is DEFERRED to S6 on real equity data (synthetic independent-
noise fixtures are not a calibration oracle).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity.forecaster._splits import rolling_ts_folds
from equity.forecaster.config import QuantileHeadParams
from equity.forecaster.quantile import QuantileHeadForecaster, _quantile_col
from equity.forecaster.targets import add_forward_targets


def _panel(n: int = 90, n_tickers: int = 5) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01 16:00", periods=n, freq="D", tz="UTC")
    sectors = ["Tech", "Tech", "Fin", "Fin", "Energy"]
    frames = []
    for t in range(n_tickers):
        rng = np.random.default_rng(40 + t)
        # Mild drift + noise so quantile heads have something to fit.
        close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": f"TK{t}",
                    "period_close_ts": ts,
                    "close": close,
                    "sector": sectors[t],
                    "f_a": rng.normal(0, 1, n),
                    "f_b": rng.normal(0, 1, n),
                    "f_c": rng.normal(0, 1, n),
                    "volume": rng.integers(1_000, 8_000, n).astype(float),
                }
            )
        )
    return add_forward_targets(
        pd.concat(frames, ignore_index=True), horizons=(1, 5, 10)
    )


def _ordered(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(
        ["period_close_ts", "ticker"], kind="mergesort"
    ).reset_index(drop=True)


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel()


def test_quantile_col_naming() -> None:
    assert _quantile_col(5, 0.05) == "pred_h5_q05"
    assert _quantile_col(1, 0.5) == "pred_h1_q50"
    assert _quantile_col(21, 0.95) == "pred_h21_q95"


def test_quantile_fit_predict_columns(panel: pd.DataFrame) -> None:
    n_folds = 3
    ordered = _ordered(panel)
    head = QuantileHeadForecaster(
        horizons=(1, 5),
        quantiles=(0.05, 0.5, 0.95),
        params=QuantileHeadParams(n_estimators=25, max_depth=3, min_child_samples=5),
        n_folds=n_folds,
        seed=9,
    )
    head.fit(ordered, return_oof=True)
    oof = head.oof_predictions().reset_index(drop=True)
    folds = rolling_ts_folds(ordered, time_col="period_close_ts", n_folds=n_folds)
    val_union = set()
    for _tr, va in folds:
        val_union.update(int(i) for i in va)

    for h in (1, 5):
        y = ordered[f"ret_h{h}"]
        labeled_val = {i for i in val_union if pd.notna(y.iloc[i])}
        for q in (0.05, 0.5, 0.95):
            col = _quantile_col(h, q)
            assert col in oof.columns
            finite_idx = {
                int(i)
                for i in np.where(np.isfinite(oof[col].to_numpy(dtype=float)))[0]
            }
            # Val-fold coverage only (earliest block stays NaN).
            assert finite_idx == labeled_val
    # 15 would be full H×Q; here 2×3 = 6 models.
    assert len(head.models_) == 6


def test_quantile_ordering_q05_le_q50_le_q95(panel: pd.DataFrame) -> None:
    head = QuantileHeadForecaster(
        horizons=(1,),
        quantiles=(0.05, 0.5, 0.95),
        params=QuantileHeadParams(n_estimators=30, max_depth=3, min_child_samples=5),
        n_folds=3,
        seed=5,
    )
    head.fit(panel, return_oof=False)
    preds = head.predict(panel)
    # Soft: most rows should satisfy monotone quantiles (LightGBM quantile
    # heads are independent so crossings can happen; require majority).
    lo = preds["pred_h1_q05"].to_numpy()
    mid = preds["pred_h1_q50"].to_numpy()
    hi = preds["pred_h1_q95"].to_numpy()
    ok = (lo <= mid + 1e-9) & (mid <= hi + 1e-9)
    assert ok.mean() >= 0.7, f"quantile crossings too frequent: {1 - ok.mean():.2%}"


def test_quantile_heldout_coverage_band(panel: pd.DataFrame) -> None:
    """Empirical 90% coverage on held-out ts-group fold — two-sided sanity band.

    Strict calibration band [0.85, 0.95] is DEFERRED to S6 on real equity data.
    Synthetic independent-noise fixtures are not a calibration oracle; we only
    catch broken (too low) and degenerate-∞ (c≈1.0) heads.
    """
    head = QuantileHeadForecaster(
        horizons=(1,),
        quantiles=(0.05, 0.5, 0.95),
        params=QuantileHeadParams(n_estimators=40, max_depth=3, min_child_samples=5),
        n_folds=3,
        seed=42,
    )
    cov = head.heldout_coverage(panel, test_frac=0.25)
    assert 1 in cov
    c = cov[1]
    assert np.isfinite(c), c
    assert 0.0 <= c <= 1.0
    # Two-sided soft band: catch c~0 (broken) and c~1.0 (degenerate wide bands).
    assert 0.6 <= c <= 0.98, f"coverage outside soft band: {c}"
