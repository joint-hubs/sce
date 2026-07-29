"""
@module: tests.equity.test_horizon_coverage
@depends: equity.forecaster.{sector_head,residual,quantile}
@exports:
@data_flow: long panel -> all three heads with horizons incl. h=21/63

FOC-52 R2: exercise .fit() / OOF / predict on the full horizon set including
h=21 and h=63 (panels used in unit tests are too short for those labels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity.forecaster._splits import rolling_ts_folds
from equity.forecaster.config import (
    QuantileHeadParams,
    ResidualHeadParams,
    SectorHeadParams,
)
from equity.forecaster.quantile import QuantileHeadForecaster, _quantile_col
from equity.forecaster.residual import InstrumentResidualForecaster
from equity.forecaster.sector_head import SectorHeadForecaster
from equity.forecaster.targets import add_forward_targets

HORIZONS = (1, 5, 10, 21, 63)


def _build_long_feature_panel(long_panel: pd.DataFrame) -> pd.DataFrame:
    """Attach sector + simple past features + forward targets on a long OHLCV panel."""
    sectors = {
        "TK0": "Tech",
        "TK1": "Tech",
        "TK2": "Fin",
        "TK3": "Energy",
        "TK4": "Health",
    }
    frames = []
    for ticker, g in long_panel.groupby("ticker", sort=False):
        g = g.sort_values("period_close_ts").copy()
        close = g["close"].to_numpy(dtype=float)
        ret1 = np.log(close / np.roll(close, 1))
        ret1[0] = np.nan
        rng = np.random.default_rng(abs(hash(str(ticker))) % (2**31))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": g["ticker"].to_numpy(),
                    "period_close_ts": g["period_close_ts"].to_numpy(),
                    "close": close,
                    "sector": sectors.get(str(ticker), "unknown"),
                    "f_mom": ret1,
                    "f_vol": rng.normal(0, 1, len(g)),
                    "volume": g["volume"].to_numpy(dtype=float),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    return add_forward_targets(panel, horizons=HORIZONS)


@pytest.fixture(scope="module")
def long_feature_panel(long_panel: pd.DataFrame) -> pd.DataFrame:
    ordered = _build_long_feature_panel(long_panel)
    return ordered.sort_values(
        ["period_close_ts", "ticker"], kind="mergesort"
    ).reset_index(drop=True)


def test_all_heads_fit_h21_h63(long_feature_panel: pd.DataFrame) -> None:
    """Fit sector + residual + quantile on full H incl. 21/63; assert OOF + predict."""
    panel = long_feature_panel
    n_folds = 4
    # Labels exist for h=63 on early portion (~n - 63 days/ticker).
    assert panel["ret_h21"].notna().sum() > 100
    assert panel["ret_h63"].notna().sum() > 100

    sector = SectorHeadForecaster(
        horizons=HORIZONS,
        params=SectorHeadParams(n_estimators=20, max_depth=3),
        n_folds=n_folds,
        seed=13,
    )
    sector.fit(panel, return_oof=True)
    oof_s = sector.oof_predictions().reset_index(drop=True)

    residual = InstrumentResidualForecaster(
        horizons=HORIZONS,
        params=ResidualHeadParams(n_estimators=20, max_depth=3),
        n_folds=n_folds,
        seed=13,
    )
    residual.fit(panel, sector_preds=oof_s, return_oof=True)
    oof_r = residual.oof_predictions().reset_index(drop=True)

    quantile = QuantileHeadForecaster(
        horizons=HORIZONS,
        quantiles=(0.05, 0.5, 0.95),
        params=QuantileHeadParams(n_estimators=20, max_depth=3, min_child_samples=5),
        n_folds=n_folds,
        seed=13,
    )
    quantile.fit(panel, return_oof=True)
    oof_q = quantile.oof_predictions().reset_index(drop=True)

    folds = rolling_ts_folds(panel, time_col="period_close_ts", n_folds=n_folds)
    val_union = set()
    for _tr, va in folds:
        val_union.update(int(i) for i in va)
    unique_ts = pd.Index(panel["period_close_ts"].drop_duplicates().sort_values())
    max_feasible_test = max(1, len(unique_ts) // (n_folds + 1))
    earliest_ts = set(unique_ts[:max_feasible_test])
    earliest_rows = {
        int(i)
        for i in panel.index[panel["period_close_ts"].isin(earliest_ts)].to_numpy()
    }

    for h in (21, 63):
        y = panel[f"ret_h{h}"]
        labeled_val = {i for i in val_union if pd.notna(y.iloc[i])}
        assert len(labeled_val) > 0, f"no labeled val rows for h={h}"

        scol = f"pred_sector_h{h}"
        finite_s = {
            int(i)
            for i in np.where(np.isfinite(oof_s[scol].to_numpy(dtype=float)))[0]
        }
        assert finite_s == labeled_val
        assert earliest_rows.isdisjoint(finite_s)

        rcol = f"pred_h{h}"
        finite_r = {
            int(i)
            for i in np.where(np.isfinite(oof_r[rcol].to_numpy(dtype=float)))[0]
        }
        # Residual coverage ⊆ sector val coverage (first residual fold may skip
        # when its train is the sector-NaN earliest block — subset is OK).
        assert finite_r.issubset(finite_s)
        assert len(finite_r) > 0
        assert earliest_rows.isdisjoint(finite_r)

        qcol = _quantile_col(h, 0.5)
        finite_q = {
            int(i)
            for i in np.where(np.isfinite(oof_q[qcol].to_numpy(dtype=float)))[0]
        }
        assert finite_q == labeled_val
        assert earliest_rows.isdisjoint(finite_q)

    # Predict on the last ~15% timestamps — full-fit models must emit finite H.
    unique_ts_sorted = panel["period_close_ts"].drop_duplicates().sort_values()
    split_day = unique_ts_sorted.iloc[int(len(unique_ts_sorted) * 0.85)]
    test = panel.loc[panel["period_close_ts"] >= split_day].copy()
    assert len(test) > 0

    sp = sector.predict(test)
    rp = residual.predict(test, sector_preds=sp)
    qp = quantile.predict(test)
    for h in (21, 63):
        assert np.isfinite(sp[f"pred_sector_h{h}"].to_numpy()).all()
        assert np.isfinite(rp[f"pred_h{h}"].to_numpy()).all()
        assert np.isfinite(qp[_quantile_col(h, 0.5)].to_numpy()).all()
