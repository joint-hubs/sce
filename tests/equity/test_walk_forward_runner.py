"""
@module: tests.equity.test_walk_forward_runner
@depends: equity.forecaster.run_walk_forward, equity.forecaster._splits
@exports:
@data_flow: synthetic short panel -> walk_forward_folds / run_walk_forward
            -> fold predictions + metadata schema

S6.1/S6.2 walk-forward runner tests. Shortened geometry (NOT the 5y defaults).
Uses ``sce_enrich=False`` so CI stays fast; SCE path is covered elsewhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity.diagnostics.walk_forward_monotonicity import run_walk_forward_monotonicity
from equity.forecaster._splits import WalkForwardFold, walk_forward_folds
from equity.forecaster.config import (
    QuantileHeadParams,
    ResidualHeadParams,
    SectorHeadParams,
    WalkForwardConfig,
)
from equity.forecaster.run_walk_forward import run_walk_forward


def _prices(n: int = 90, n_tickers: int = 4) -> pd.DataFrame:
    """Synthetic multi-ticker OHLCV panel long enough for >=2 short WF folds."""
    ts = pd.date_range("2024-01-01 16:00", periods=n, freq="D", tz="America/New_York")
    frames = []
    for t in range(n_tickers):
        rng = np.random.default_rng(50 + t)
        close = 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
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
                    "volume": rng.integers(1000, 9000, n).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _sectors(n_tickers: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [f"TK{t}" for t in range(n_tickers)],
            "sector": ["Tech", "Tech", "Fin", "Energy"][:n_tickers],
        }
    )


def _short_cfg(**overrides) -> WalkForwardConfig:
    base = dict(
        train_window=30,
        val_window=8,
        test_window=8,
        step=8,
        horizons=(1, 5),
        quantiles=(0.05, 0.5, 0.95),
        seed=0,
        run_grade="diagnostic",
        n_folds=3,
        sector=SectorHeadParams(n_estimators=12, max_depth=2),
        residual=ResidualHeadParams(n_estimators=12, max_depth=2),
        quantile=QuantileHeadParams(n_estimators=12, max_depth=2, min_child_samples=5),
    )
    base.update(overrides)
    return WalkForwardConfig(**base)


def test_walk_forward_config_rejects_bad_windows() -> None:
    with pytest.raises(ValueError, match="step"):
        WalkForwardConfig(step=0)
    with pytest.raises(ValueError, match="train_window"):
        WalkForwardConfig(train_window=0)


def test_walk_forward_folds_count_and_bounds() -> None:
    # n_ts=90, block=30+8+8=46, step=8 → starts 0,8,16,24,32,40,48 → 48+46=94>90 so 6 folds
    # 0..5 = 6 folds (last start=40: 40+46=86<=90; start=48: 48+46=94>90 stop)
    prices = _prices(n=90, n_tickers=3)
    cfg = _short_cfg()
    folds = walk_forward_folds(prices, cfg)
    assert len(folds) >= 2
    # expected exact count
    block = cfg.train_window + cfg.val_window + cfg.test_window
    expected = 1 + (90 - block) // cfg.step
    assert len(folds) == expected

    for f in folds:
        assert isinstance(f, WalkForwardFold)
        assert f.train_end < f.val_start <= f.val_end < f.test_start <= f.test_end

    mono = run_walk_forward_monotonicity(
        [f.to_monotonicity_record() for f in folds], strict=False
    )
    assert mono["pass"] is True


def test_walk_forward_folds_too_short_raises() -> None:
    prices = _prices(n=20, n_tickers=2)
    with pytest.raises(ValueError, match="need >="):
        walk_forward_folds(prices, _short_cfg(train_window=30, val_window=8, test_window=8))


@pytest.fixture(scope="module")
def wf_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("wf")
    # n=70 → block=46 → starts 0,8,16,24 → 4 folds. Keep runtime modest.
    cfg = _short_cfg()
    result = run_walk_forward(
        _prices(n=70, n_tickers=4),
        features=None,
        sectors=_sectors(4),
        cfg=cfg,
        out_dir=out,
        sce_enrich=False,
        git_sha="deadbeef",
        created_at="2026-07-29T00:00:00Z",
    )
    return result


def test_wf_fold_count(wf_result) -> None:
    # n=70, block=46, step=8 → starts 0,8,16,24 → 4
    assert wf_result["n_folds"] == 4
    assert len(wf_result["fold_predictions_test"]) == 4
    assert len(wf_result["fold_predictions_val"]) == 4


def test_wf_monotonicity_pass(wf_result) -> None:
    assert wf_result["monotonicity"]["pass"] is True


def test_wf_predictions_finite_and_schema(wf_result) -> None:
    for fold_map in wf_result["fold_predictions_test"]:
        for h, df in fold_map.items():
            assert f"pred_h{h}" in df.columns
            assert f"pred_sector_h{h}" in df.columns
            assert f"pred_resid_h{h}" in df.columns
            assert f"pred_h{h}_q05" in df.columns
            assert f"pred_h{h}_q50" in df.columns
            assert f"pred_h{h}_q95" in df.columns
            assert "split" in df.columns
            assert (df["split"] == "test").all()
            # Predictions should be finite (models always emit a number).
            assert np.isfinite(df[f"pred_h{h}"].to_numpy(dtype=float)).all()
            assert np.isfinite(df[f"pred_sector_h{h}"].to_numpy(dtype=float)).all()


def test_wf_artifacts_on_disk(wf_result) -> None:
    out = Path(wf_result["out_dir"])
    assert (out / "metadata.json").is_file()
    for k in range(wf_result["n_folds"]):
        fold_dir = out / f"fold_{k:02d}"
        assert fold_dir.is_dir()
        for h in (1, 5):
            path = fold_dir / f"predictions_h{h}.parquet"
            assert path.is_file(), path
            df = pd.read_parquet(path)
            # val + test combined, tagged via split
            assert set(df["split"].unique()) == {"val", "test"}
            assert f"pred_h{h}" in df.columns


def test_wf_metadata_schema(wf_result) -> None:
    meta = wf_result["metadata"]
    for key in (
        "git_sha",
        "config_hash",
        "seed",
        "run_grade",
        "horizons",
        "quantiles",
        "created_at",
        "walk_forward",
        "metrics",
    ):
        assert key in meta, key
    assert meta["git_sha"] == "deadbeef"
    assert meta["run_grade"] == "diagnostic"
    assert meta["horizons"] == [1, 5]

    wf = meta["walk_forward"]
    for key in (
        "n_folds",
        "train_window",
        "val_window",
        "test_window",
        "step",
        "fold_bounds",
        "monotonicity",
    ):
        assert key in wf, key
    assert wf["n_folds"] == 4
    assert wf["train_window"] == 30
    assert wf["monotonicity"]["pass"] is True
    assert len(wf["fold_bounds"]) == 4

    metrics = meta["metrics"]
    assert "1" in metrics and "5" in metrics
    for hkey in ("1", "5"):
        m = metrics[hkey]
        for k in (
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mae_std",
            "hit_rate_mean",
            "hit_rate_std",
        ):
            assert k in m, k
            # JSON-safe: must not be NaN token (None allowed)
            if m[k] is not None:
                assert isinstance(m[k], (int, float))


def test_wf_result_metrics_dict(wf_result) -> None:
    # in-memory metrics keyed by int horizon
    assert 1 in wf_result["metrics"]
    assert 5 in wf_result["metrics"]
    m = wf_result["metrics"][1]
    assert "rmse_mean" in m
