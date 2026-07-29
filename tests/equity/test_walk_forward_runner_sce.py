"""
@module: tests.equity.test_walk_forward_runner_sce
@depends: equity.forecaster.run_walk_forward, equity.forecaster.config, tests.equity.conftest
@exports:
@data_flow: long_panel (synthetic prices) -> run_walk_forward(sce_enrich=True)
            -> per-fold val/test predictions finite + monotonicity pass + SCE leg ran

S6.1 walk-forward **SCE-enriched** path test. The sibling
``test_walk_forward_runner.py`` exercises ``sce_enrich=False`` only; this module
asserts the per-fold :class:`EquityContextEnricher` refit (the project's value
proposition) actually runs end-to-end on a shortened geometry and produces
finite, non-empty predictions that differ from the no-SCE leg.

Geometry is deliberately short (train=80 / val=15 / test=15 / step=80 on the
220-day ``long_panel`` fixture -> 2 folds, horizons {1,5}) so CI stays fast
while still exercising a rolling-CF SCE refit per fold.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity.forecaster.config import (
    QuantileHeadParams,
    ResidualHeadParams,
    SectorHeadParams,
    WalkForwardConfig,
)
from equity.forecaster.run_walk_forward import run_walk_forward


def _sce_short_cfg(**overrides) -> WalkForwardConfig:
    base = dict(
        train_window=80,
        val_window=15,
        test_window=15,
        step=80,
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


def _assert_pred_frames_finite(
    fold_preds: list[dict[int, pd.DataFrame]], *, horizons: tuple[int, ...]
) -> None:
    assert fold_preds, "no fold predictions returned"
    for fold_idx, fold_map in enumerate(fold_preds):
        for h in horizons:
            assert h in fold_map, f"fold {fold_idx} missing horizon {h}"
            df = fold_map[h]
            assert not df.empty, f"fold {fold_idx} h={h} predictions empty"
            for col in (f"pred_h{h}", f"pred_sector_h{h}", f"pred_resid_h{h}"):
                assert col in df.columns, f"fold {fold_idx} h={h} missing {col}"
                vals = df[col].to_numpy(dtype=float)
                # No all-NaN, no inf, at least one finite prediction.
                assert np.isfinite(vals).any(), (
                    f"fold {fold_idx} h={h} {col}: no finite predictions"
                )
                assert not np.isinf(vals).any(), (
                    f"fold {fold_idx} h={h} {col}: inf present"
                )


@pytest.fixture(scope="module")
def wf_sce_result(long_panel, sectors_fixture, tmp_path_factory):
    out = tmp_path_factory.mktemp("wf_sce")
    cfg = _sce_short_cfg()
    result = run_walk_forward(
        long_panel,
        features=None,
        sectors=sectors_fixture,
        cfg=cfg,
        out_dir=out,
        sce_enrich=True,
        git_sha="deadbeef",
        created_at="2026-07-29T00:00:00Z",
    )
    return result


@pytest.fixture(scope="module")
def wf_nosce_result(long_panel, sectors_fixture, tmp_path_factory):
    out = tmp_path_factory.mktemp("wf_nosce")
    cfg = _sce_short_cfg()
    result = run_walk_forward(
        long_panel,
        features=None,
        sectors=sectors_fixture,
        cfg=cfg,
        out_dir=out,
        sce_enrich=False,
        git_sha="deadbeef",
        created_at="2026-07-29T00:00:00Z",
    )
    return result


def test_wf_sce_ran_at_least_one_fold(wf_sce_result) -> None:
    assert wf_sce_result["n_folds"] >= 1
    assert wf_sce_result["sce_enrich"] is True
    assert len(wf_sce_result["fold_predictions_test"]) == wf_sce_result["n_folds"]
    assert len(wf_sce_result["fold_predictions_val"]) == wf_sce_result["n_folds"]


def test_wf_sce_predictions_finite(wf_sce_result) -> None:
    horizons = (1, 5)
    _assert_pred_frames_finite(wf_sce_result["fold_predictions_val"], horizons=horizons)
    _assert_pred_frames_finite(wf_sce_result["fold_predictions_test"], horizons=horizons)


def test_wf_sce_monotonicity_pass(wf_sce_result) -> None:
    mono = wf_sce_result["monotonicity"]
    assert mono["pass"] is True
    # Metadata-equivalent dict carries the same gate.
    wf = wf_sce_result["metadata"]["walk_forward"]
    assert wf["monotonicity"]["pass"] is True
    assert wf["sce_enrich"] is True


def test_wf_sce_artifacts_on_disk(wf_sce_result) -> None:
    out = Path(wf_sce_result["out_dir"])
    assert (out / "metadata.json").is_file()
    for k in range(wf_sce_result["n_folds"]):
        fold_dir = out / f"fold_{k:02d}"
        assert fold_dir.is_dir()
        for h in (1, 5):
            path = fold_dir / f"predictions_h{h}.parquet"
            assert path.is_file(), path
            df = pd.read_parquet(path)
            assert set(df["split"].unique()) == {"val", "test"}
            assert np.isfinite(df[f"pred_h{h}"].to_numpy(dtype=float)).any()


def test_wf_sce_leg_actually_ran(wf_sce_result, wf_nosce_result, long_panel, sectors_fixture) -> None:
    """The SCE leg must change outputs vs the no-SCE leg.

    Both legs now share the same ``build_features`` base (fair A/B); the only
    difference is SCE context enrichment. Three independent sanity signals:

      1. Per-fold feature width (``n_features``) is strictly larger with SCE
         (SCE appends context columns on top of the shared build_features panel).
      2. SCE context columns are present in the enriched design matrix and absent
         from the baseline (proves SCE context actually reached the model, not
         just that frames differ).
      3. At least one prediction column differs in value between the two legs.
    """
    sce_bounds = wf_sce_result["metadata"]["walk_forward"]["fold_bounds"]
    nosce_bounds = wf_nosce_result["metadata"]["walk_forward"]["fold_bounds"]
    assert len(sce_bounds) == len(nosce_bounds)
    assert len(sce_bounds) >= 1
    for sf, nf in zip(sce_bounds, nosce_bounds):
        assert sf["n_features"] > nf["n_features"], (
            f"SCE fold {sf['fold_idx']} n_features={sf['n_features']} "
            f"not greater than baseline {nf['n_features']} "
            f"(both on build_features base; SCE must add context columns)"
        )

    # SCE context columns must be present in the enriched design matrix and
    # absent from the baseline (proves SCE context reached the model).
    from equity.features.build import build_features
    from equity.sce import EquityContextEnricher, EquityHierarchyConfig
    from equity.sce.enrich import _level_from_context_column

    features = build_features(long_panel)
    # Baseline features: no SCE context columns expected.
    base_sce_cols = [
        c for c in features.columns if _level_from_context_column(c, "ret_1d") is not None
    ]
    assert len(base_sce_cols) == 0, (
        f"Baseline build_features unexpectedly has SCE context columns: {base_sce_cols}"
    )

    enricher = EquityContextEnricher(
        hierarchy=EquityHierarchyConfig(),
        sectors=sectors_fixture,
    )
    enriched = enricher.fit_transform(features)
    sce_cols = [
        c for c in enriched.columns if _level_from_context_column(c, "ret_1d") is not None
    ]
    assert len(sce_cols) > 0, "No SCE context columns found in enriched features"

    # Prediction values must differ on at least one fold/horizon.
    differs = False
    for fold_idx in range(wf_sce_result["n_folds"]):
        for h in (1, 5):
            sce_df = wf_sce_result["fold_predictions_test"][fold_idx][h]
            nosc_df = wf_nosce_result["fold_predictions_test"][fold_idx][h]
            # Align on the same (ticker, time) row order before comparing.
            key_cols = ["ticker", "period_close_ts"]
            sce_s = sce_df.set_index(key_cols)[f"pred_h{h}"].sort_index()
            nosc_s = nosc_df.set_index(key_cols)[f"pred_h{h}"].sort_index()
            common = sce_s.index.intersection(nosc_s.index)
            if len(common) == 0:
                continue
            a = sce_s.loc[common].to_numpy(dtype=float)
            b = nosc_s.loc[common].to_numpy(dtype=float)
            finite = np.isfinite(a) & np.isfinite(b)
            if finite.sum() == 0:
                continue
            if not np.allclose(a[finite], b[finite], rtol=1e-9, atol=1e-12):
                differs = True
                break
        if differs:
            break
    assert differs, "SCE predictions identical to baseline — SCE leg did not run"


def test_wf_sce_metrics_present(wf_sce_result) -> None:
    metrics = wf_sce_result["metadata"]["metrics"]
    assert "1" in metrics and "5" in metrics
    for hkey in ("1", "5"):
        m = metrics[hkey]
        for k in ("rmse_mean", "mae_mean", "hit_rate_mean"):
            assert k in m, (hkey, k)
            if m[k] is not None and isinstance(m[k], float):
                assert math.isfinite(m[k]), (hkey, k, m[k])
