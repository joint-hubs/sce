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

from equity.forecaster._splits import rolling_ts_folds
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


def _ordered(panel: pd.DataFrame) -> pd.DataFrame:
    return panel.sort_values(
        ["period_close_ts", "ticker"], kind="mergesort"
    ).reset_index(drop=True)


def _val_union_and_earliest_block(
    ordered: pd.DataFrame, *, n_folds: int, time_col: str = "period_close_ts"
) -> tuple[set[int], set[int]]:
    """Union of val-row positions across folds + earliest train-only block rows."""
    folds = rolling_ts_folds(ordered, time_col=time_col, n_folds=n_folds)
    val_union: set[int] = set()
    for _tr, va in folds:
        val_union.update(int(i) for i in va)
    unique_ts = pd.Index(ordered[time_col].drop_duplicates().sort_values())
    n_ts = len(unique_ts)
    max_feasible_test = max(1, n_ts // (n_folds + 1))
    earliest_ts = set(unique_ts[:max_feasible_test])
    earliest_rows = {
        int(i) for i in ordered.index[ordered[time_col].isin(earliest_ts)].to_numpy()
    }
    return val_union, earliest_rows


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return _panel()


def test_sector_head_oof_valfold_coverage(panel: pd.DataFrame) -> None:
    """OOF finite exactly on val-fold rows (earliest train-only block is NaN)."""
    horizons = (1, 5, 10)
    n_folds = 4
    ordered = _ordered(panel)
    head = SectorHeadForecaster(
        horizons=horizons,
        params=SectorHeadParams(n_estimators=20, max_depth=3),
        n_folds=n_folds,
        seed=7,
    )
    head.fit(ordered, return_oof=True)
    oof = head.oof_predictions().reset_index(drop=True)
    val_union, earliest_rows = _val_union_and_earliest_block(ordered, n_folds=n_folds)

    for h in horizons:
        col = f"pred_sector_h{h}"
        assert col in oof.columns
        y = ordered[f"ret_h{h}"]
        labeled_val = {i for i in val_union if pd.notna(y.iloc[i])}
        finite_idx = {
            int(i) for i in np.where(np.isfinite(oof[col].to_numpy(dtype=float)))[0]
        }
        assert finite_idx == labeled_val, (
            f"{col}: finite OOF set != labeled val-union "
            f"(extra={finite_idx - labeled_val}, missing={labeled_val - finite_idx})"
        )
        # Earliest train-only block must stay NaN (no future-trained fill).
        for i in earliest_rows:
            assert not np.isfinite(oof[col].iloc[i]), (
                f"{col}: earliest train-only row {i} unexpectedly finite"
            )


def test_sector_head_oof_temporal_integrity(panel: pd.DataFrame) -> None:
    """Prove no future-trained fill: finite OOF ⊆ val union; train strictly earlier.

    Re-introducing a future-trained filler on the earliest block expands the
    finite set beyond val-union and fails this test.
    """
    horizons = (1, 5)
    n_folds = 4
    ordered = _ordered(panel)
    head = SectorHeadForecaster(
        horizons=horizons,
        params=SectorHeadParams(n_estimators=20, max_depth=3),
        n_folds=n_folds,
        seed=7,
    )
    head.fit(ordered, return_oof=True)
    oof = head.oof_predictions().reset_index(drop=True)
    folds = rolling_ts_folds(ordered, time_col="period_close_ts", n_folds=n_folds)
    val_union, earliest_rows = _val_union_and_earliest_block(ordered, n_folds=n_folds)

    # Every val row is associated with a fold whose train is strictly earlier.
    for _tr, va in folds:
        if len(va) == 0 or len(_tr) == 0:
            continue
        assert ordered.iloc[_tr]["period_close_ts"].max() < ordered.iloc[va][
            "period_close_ts"
        ].min()

    for h in horizons:
        col = f"pred_sector_h{h}"
        y = ordered[f"ret_h{h}"]
        labeled_val = {i for i in val_union if pd.notna(y.iloc[i])}
        finite_idx = {
            int(i) for i in np.where(np.isfinite(oof[col].to_numpy(dtype=float)))[0]
        }
        assert finite_idx == labeled_val
        assert earliest_rows.isdisjoint(finite_idx)
        assert len(finite_idx) > 0


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
