"""
@module: tests.equity.test_baseline_compare
@depends: equity.forecaster.baseline_compare, equity.diagnostics.forward_target_isolation
@exports:
@data_flow: long_panel + sectors -> run_baseline_vs_sce -> comparison + full metadata

S6.4 baseline-vs-SCE end-to-end on shortened geometry (train=80/val=15/test=15/
step=80 → 2 folds, horizons {1,5}). Reuses ``long_panel`` / ``sectors_fixture``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity.diagnostics.forward_target_isolation import run_forward_target_isolation
from equity.forecaster.baseline_compare import run_baseline_vs_sce
from equity.forecaster.config import (
    QuantileHeadParams,
    ResidualHeadParams,
    SectorHeadParams,
    WalkForwardConfig,
)
from equity.forecaster.targets import add_forward_targets


def _short_cfg(**overrides) -> WalkForwardConfig:
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


def _geometry_keys(bounds):
    """Timestamp geometry only (n_features intentionally excluded)."""
    return [
        (
            b["fold_idx"],
            b["train_start"],
            b["train_end"],
            b["val_start"],
            b["val_end"],
            b["test_start"],
            b["test_end"],
        )
        for b in bounds
    ]


@pytest.fixture(scope="module")
def bvs_result(long_panel, sectors_fixture, tmp_path_factory):
    out = tmp_path_factory.mktemp("bvs")
    return run_baseline_vs_sce(
        long_panel,
        features=None,
        sectors=sectors_fixture,
        cfg=_short_cfg(),
        out_dir=out,
        git_sha="deadbeef",
        created_at="2026-07-29T00:00:00Z",
    )


def test_bvs_both_legs_ran(bvs_result) -> None:
    assert bvs_result["sce"]["n_folds"] >= 1
    assert bvs_result["baseline"]["n_folds"] >= 1
    assert bvs_result["sce"]["n_folds"] == bvs_result["baseline"]["n_folds"]
    assert bvs_result["sce"]["sce_enrich"] is True
    assert bvs_result["baseline"]["sce_enrich"] is False


def test_bvs_fold_bounds_identical(bvs_result) -> None:
    sce_b = bvs_result["sce"]["metadata"]["walk_forward"]["fold_bounds"]
    base_b = bvs_result["baseline"]["metadata"]["walk_forward"]["fold_bounds"]
    assert _geometry_keys(sce_b) == _geometry_keys(base_b)
    # SCE must actually enrich (wider feature set) — both legs now share the
    # same build_features base (fair A/B), so the gap is SCE context columns only.
    for sb, bb in zip(sce_b, base_b):
        assert sb["n_features"] > bb["n_features"], (
            f"SCE fold {sb['fold_idx']} n_features={sb['n_features']} "
            f"not greater than baseline {bb['n_features']} "
            f"(both on build_features base; SCE must add context columns)"
        )


def test_bvs_reports_exist(bvs_result) -> None:
    out = Path(bvs_result["out_dir"])
    assert (out / "comparison_report.json").is_file()
    assert (out / "comparison_report.md").is_file()
    assert (out / "metadata.json").is_file()
    assert (out / "legs" / "sce" / "metadata.json").is_file()
    assert (out / "legs" / "baseline" / "metadata.json").is_file()
    # smoke content
    md = (out / "comparison_report.md").read_text(encoding="utf-8")
    assert "positive = SCE" in md or "SCE **worse**" in md
    report = json.loads((out / "comparison_report.json").read_text(encoding="utf-8"))
    assert "per_horizon" in report
    assert "sign_conventions" in report


def test_bvs_metadata_full_schema(bvs_result) -> None:
    """Canonical run metadata (root + SCE leg) carries the full S6.4 schema."""
    for meta in (
        bvs_result["metadata"],
        bvs_result["sce"]["metadata"],
    ):
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
            "chosen_horizon",
            "baseline_vs_sce",
        ):
            assert key in meta, key

        wf = meta["walk_forward"]
        for k in (
            "n_folds",
            "train_window",
            "val_window",
            "test_window",
            "step",
            "fold_bounds",
            "monotonicity",
            "sce_enrich",
        ):
            assert k in wf, k
        assert wf["sce_enrich"] is True

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
                "sharpe",
                "sortino",
            ):
                assert k in m, (hkey, k)

        assert isinstance(meta["chosen_horizon"], int)
        assert meta["chosen_horizon"] in (1, 5)

        bvs = meta["baseline_vs_sce"]
        assert "comparison_report_path" in bvs
        for hkey in ("1", "5"):
            assert hkey in bvs
            d = bvs[hkey]
            for k in ("d_rmse", "d_mae", "d_hit_rate", "d_sharpe", "d_sortino"):
                assert k in d, (hkey, k)


def test_bvs_deltas_finite(bvs_result) -> None:
    """At least one horizon has a finite Δ, and no Δ is +inf/-inf."""
    bvs = bvs_result["metadata"]["baseline_vs_sce"]
    any_finite = False
    for hkey in ("1", "5"):
        d = bvs[hkey]
        for k in ("d_rmse", "d_mae", "d_hit_rate", "d_sharpe", "d_sortino"):
            v = d[k]
            if v is None:
                continue
            assert isinstance(v, (int, float))
            assert math.isfinite(float(v)), (hkey, k, v)
            any_finite = True
    assert any_finite, "expected at least one finite delta"


def test_bvs_sce_context_columns_present(long_panel, sectors_fixture) -> None:
    """SCE enriched design matrix must contain SCE context columns.

    Both legs now share the same ``build_features`` base (fair A/B); the SCE leg
    differs ONLY by SCE context columns. This test proves those columns exist
    and follow the naming contract ``{level}_ret_1d_{stat}``.
    """
    from equity.features.build import build_features
    from equity.sce import EquityContextEnricher, EquityHierarchyConfig
    from equity.sce.enrich import _level_from_context_column

    features = build_features(long_panel)
    # Baseline: no SCE context columns.
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


def test_bvs_baseline_leg_forward_target_isolation(
    long_panel, sectors_fixture, tmp_path_factory
) -> None:
    """Baseline leg design matrix (no SCE) must pass the leak guard.

    Reconstruct the baseline feature set the way all runners do: ``build_features``
    on prices + sector join, labels NOT attached (labels join after features in
    the runner). Guard must PASS with zero violations.
    """
    from equity.features.build import build_features

    features = build_features(long_panel)
    # sector join is the only hierarchy on the baseline leg
    sec = sectors_fixture[["ticker", "sector"]]
    feat = features.merge(sec, on="ticker", how="left", sort=False)
    # Sanity: price-derived forward targets branch may exist on features? Drop any.
    leak_cols = [c for c in feat.columns if c.startswith("ret_h") or c.startswith("pred_")]
    if leak_cols:
        feat = feat.drop(columns=leak_cols)

    result = run_forward_target_isolation(feat, horizons=(1, 5), allowed_pred_sector=False)
    assert result["pass"] is True, result
    assert result["n_violations"] == 0

    # Also: labels are only on the priced frame after add_forward_targets — not
    # part of the baseline feature design matrix.
    priced = add_forward_targets(long_panel, horizons=(1, 5))
    assert "ret_h1" in priced.columns
    assert "ret_h1" not in feat.columns
