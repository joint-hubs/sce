"""
@module: tests.equity.test_sector_head
@depends: equity.forecaster.sector_head
@exports:
@data_flow: synthetic panel + ret_hN -> SectorHeadForecaster -> OOF pred_sector_hN

S5.1 sector-head tests. Structural assertions only (coverage / finiteness /
column presence), NOT performance numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity.forecaster.config import SectorHeadParams
from equity.forecaster.sector_head import SectorHeadForecaster
from equity.forecaster.targets import add_forward_targets


def _panel(n: int = 80, n_tickers: int = 4) -> pd.DataFrame:
    """Minimal feature panel with sector + a couple of numeric features."""
    ts = pd.date_range("2024-01-01 16:00", periods=n, freq="D", tz="UTC")
    sectors = ["Tech", "Tech", "Fin", "Fin"]
    frames = []
    for t in range(n_tickers):
        rng = np.random.default_rng(20 + t)
        close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.015, n))
        # Simple past-only feature: 1d log return shifted.
        ret1 = np.log(close / np.roll(close, 1))
        ret1[0] = np.nan
        frames.append(
            pd.DataFrame(
                {
                    "ticker": f"TK{t}",
                    "period_close_ts": ts,
                    "close": close,
                    "sector": sectors[t],
                    "f_mom": ret1,
                    "f_vol": rng.normal(0, 1, n),
                    "volume": rng.integers(1_000, 10_000, n).astype(float),
                }
            )
        )
    prices = pd.concat(frames, ignore_index=True)
    # Horizons short enough that most labels are non-null on n=80.
    return add_forward_targets(prices, horizons=(1, 5, 10))


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel()


def test_sector_head_oof_full_labeled_coverage(panel: pd.DataFrame) -> None:
    horizons = (1, 5, 10)
    head = SectorHeadForecaster(
        horizons=horizons,
        params=SectorHeadParams(n_estimators=20, max_depth=3),
        n_folds=4,
        seed=7,
    )
    head.fit(panel, return_oof=True)
    oof = head.oof_predictions()

    for h in horizons:
        col = f"pred_sector_h{h}"
        assert col in oof.columns
        y = panel.sort_values("period_close_ts").reset_index(drop=True)[f"ret_h{h}"]
        labeled = y.notna()
        # 100% of labeled rows get an OOF prediction.
        assert labeled.sum() > 0
        assert oof.loc[labeled, col].notna().all(), (
            f"{col}: OOF coverage < 100% of labeled rows "
            f"({oof.loc[labeled, col].isna().sum()} missing of {labeled.sum()})"
        )
        assert np.isfinite(oof.loc[labeled, col].to_numpy()).all()


def test_sector_head_predict_shape(panel: pd.DataFrame) -> None:
    head = SectorHeadForecaster(
        horizons=(1, 5),
        params=SectorHeadParams(n_estimators=15, max_depth=2),
        n_folds=3,
        seed=3,
    )
    head.fit(panel.iloc[:200], return_oof=False)
    preds = head.predict(panel.iloc[200:280])
    assert len(preds) == 80
    assert "pred_sector_h1" in preds.columns
    assert "pred_sector_h5" in preds.columns
    assert np.isfinite(preds["pred_sector_h1"].to_numpy()).all()


def test_sector_head_does_not_use_forward_targets_as_features(panel: pd.DataFrame) -> None:
    head = SectorHeadForecaster(
        horizons=(1,),
        params=SectorHeadParams(n_estimators=10, max_depth=2),
        n_folds=3,
        seed=1,
    )
    head.fit(panel, return_oof=False)
    # Design matrix must never list ret_h* / pred_*.
    for c in head.feature_cols_:
        assert not c.startswith("ret_h"), c
        assert not c.startswith("pred_"), c
