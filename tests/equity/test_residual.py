"""
@module: tests.equity.test_residual
@depends: equity.forecaster.residual, equity.forecaster.sector_head
@exports:
@data_flow: panel -> sector OOF -> residual fit -> pred_hN = sector + resid

S5.2 residual-layer tests. Structural + identity of the final composition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity.forecaster.config import ResidualHeadParams, SectorHeadParams
from equity.forecaster.residual import InstrumentResidualForecaster
from equity.forecaster.sector_head import SectorHeadForecaster
from equity.forecaster.targets import add_forward_targets


def _panel(n: int = 70, n_tickers: int = 4) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01 16:00", periods=n, freq="D", tz="UTC")
    sectors = ["Tech", "Tech", "Fin", "Fin"]
    frames = []
    for t in range(n_tickers):
        rng = np.random.default_rng(30 + t)
        close = 100.0 * np.cumprod(1 + rng.normal(0.0005, 0.012, n))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": f"TK{t}",
                    "period_close_ts": ts,
                    "close": close,
                    "sector": sectors[t],
                    "f_a": rng.normal(0, 1, n),
                    "f_b": rng.normal(0, 1, n),
                    "volume": rng.integers(1_000, 5_000, n).astype(float),
                }
            )
        )
    return add_forward_targets(pd.concat(frames, ignore_index=True), horizons=(1, 5))


@pytest.fixture(scope="module")
def fitted_bundle():
    panel = _panel()
    # Match forecaster internal sort (time, ticker) so positional OOF aligns.
    ordered = panel.sort_values(
        ["period_close_ts", "ticker"], kind="mergesort"
    ).reset_index(drop=True)
    sector = SectorHeadForecaster(
        horizons=(1, 5),
        params=SectorHeadParams(n_estimators=20, max_depth=3),
        n_folds=4,
        seed=11,
    )
    sector.fit(ordered, return_oof=True)
    oof_s = sector.oof_predictions()
    resid = InstrumentResidualForecaster(
        horizons=(1, 5),
        params=ResidualHeadParams(n_estimators=20, max_depth=3),
        n_folds=4,
        seed=11,
    )
    resid.fit(ordered, sector_preds=oof_s, return_oof=True)
    # Re-read ordered from residual OOF keys to guarantee row alignment.
    oof = resid.oof_predictions().reset_index(drop=True)
    if {"ticker", "period_close_ts"}.issubset(oof.columns):
        panel_aligned = ordered.merge(
            oof[["ticker", "period_close_ts"]],
            on=["ticker", "period_close_ts"],
            how="right",
            sort=False,
        ).reset_index(drop=True)
    else:
        panel_aligned = ordered
    return {
        "panel": panel_aligned,
        "sector": sector,
        "residual": resid,
        "oof_s": oof_s,
    }


def test_residual_composition_identity(fitted_bundle) -> None:
    oof = fitted_bundle["residual"].oof_predictions()
    for h in (1, 5):
        y = fitted_bundle["panel"][f"ret_h{h}"]
        labeled = y.notna()
        # 100% labeled coverage of the final pred.
        assert oof.loc[labeled, f"pred_h{h}"].notna().all()
        # Composition identity: pred = sector + resid (where all three finite).
        both = (
            oof[f"pred_sector_h{h}"].notna()
            & oof[f"pred_resid_h{h}"].notna()
            & oof[f"pred_h{h}"].notna()
        )
        assert both.any()
        recon = (
            oof.loc[both, f"pred_sector_h{h}"].to_numpy()
            + oof.loc[both, f"pred_resid_h{h}"].to_numpy()
        )
        np.testing.assert_allclose(
            oof.loc[both, f"pred_h{h}"].to_numpy(), recon, rtol=0, atol=1e-10
        )


def test_residual_label_is_oof_based(fitted_bundle) -> None:
    """resid_hN on the OOF frame equals ret_hN - pred_sector_hN (key-aligned)."""
    oof = fitted_bundle["residual"].oof_predictions().reset_index(drop=True)
    panel = fitted_bundle["panel"]
    # Attach labels by (ticker, time) so sort differences cannot desync rows.
    labels = panel[["ticker", "period_close_ts", "ret_h1", "ret_h5"]].drop_duplicates(
        ["ticker", "period_close_ts"]
    )
    joined = oof.merge(labels, on=["ticker", "period_close_ts"], how="left", sort=False)
    assert len(joined) == len(oof)
    for h in (1, 5):
        ret = joined[f"ret_h{h}"].to_numpy(dtype=float)
        ps = joined[f"pred_sector_h{h}"].to_numpy(dtype=float)
        resid = joined[f"resid_h{h}"].to_numpy(dtype=float)
        both = np.isfinite(ret) & np.isfinite(ps) & np.isfinite(resid)
        assert both.any()
        np.testing.assert_allclose(resid[both], ret[both] - ps[both], rtol=0, atol=1e-12)


def test_residual_predict_on_heldout(fitted_bundle) -> None:
    panel = fitted_bundle["panel"]
    # Use last 20% timestamps as a pseudo test.
    unique_ts = panel["period_close_ts"].drop_duplicates().sort_values()
    split_day = unique_ts.iloc[int(len(unique_ts) * 0.8)]
    train = panel.loc[panel["period_close_ts"] < split_day].copy()
    test = panel.loc[panel["period_close_ts"] >= split_day].copy()

    sector = SectorHeadForecaster(
        horizons=(1,),
        params=SectorHeadParams(n_estimators=15, max_depth=2),
        n_folds=3,
        seed=2,
    )
    sector.fit(train, return_oof=True)
    resid = InstrumentResidualForecaster(
        horizons=(1,),
        params=ResidualHeadParams(n_estimators=15, max_depth=2),
        n_folds=3,
        seed=2,
    )
    resid.fit(train, sector_preds=sector.oof_predictions(), return_oof=False)
    preds = resid.predict(test, sector_preds=sector.predict(test))
    assert "pred_h1" in preds.columns
    assert len(preds) == len(test)
    assert np.isfinite(preds["pred_h1"].to_numpy()).all()


def test_residual_design_excludes_forward_targets(fitted_bundle) -> None:
    head = fitted_bundle["residual"]
    for c in head.feature_cols_:
        assert not str(c).startswith("ret_h"), c
        assert not str(c).startswith("pred_h"), c
        assert not str(c).startswith("resid_h"), c
