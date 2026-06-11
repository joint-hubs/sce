"""Tests for metadata reproducibility helpers."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.run import (
    _collect_run_metadata,
    _config_hash,
    _enforce_run_grade_policy,
    DatasetSplit,
)
from sce import ContextConfig
from sce.config import AggregationMethod


def test_config_hash_deterministic(tmp_path: Path):
    cfg = tmp_path / "sample.toml"
    cfg.write_text("[a]\nb = 1\n", encoding="utf-8")

    h1 = _config_hash(cfg)
    h2 = _config_hash(cfg)
    assert h1 == h2


def test_config_hash_changes_with_content(tmp_path: Path):
    cfg = tmp_path / "sample.toml"
    cfg.write_text("[a]\nb = 1\n", encoding="utf-8")
    h1 = _config_hash(cfg)

    cfg.write_text("[a]\nb = 2\n", encoding="utf-8")
    h2 = _config_hash(cfg)
    assert h1 != h2


def test_metadata_schema_complete(tmp_path: Path):
    cfg = tmp_path / "sample.toml"
    cfg.write_text("[dataset]\npath='data.parquet'\n[target]\ncolumn='price'\n", encoding="utf-8")

    df = pd.DataFrame({"price": [1.0, 2.0]})
    split = DatasetSplit(
        train_idx=pd.Index([0]),
        test_idx=pd.Index([1]),
        train_df=df.iloc[[0]],
        test_df=df.iloc[[1]],
    )
    sce_config = ContextConfig(
        target_col="price",
        categorical_cols=["city"],
        aggregations=[AggregationMethod.MEAN],
        use_cross_fitting=False,
    )

    metadata = _collect_run_metadata(
        config_name="sample",
        config_path=cfg,
        config={
            "dataset": {"source": "local", "path": "data/parquet/sample.parquet"},
            "target": {"column": "price"},
            "split": {"strategy": "random", "test_size": 0.5, "random_state": 42},
            "sce": {},
        },
        run_grade="exploratory",
        split=split,
        sce_config=sce_config,
        model_type="xgboost",
        model_params={"n_estimators": 10},
        runtime_seconds=1.5,
        metrics={
            "baseline_rmse": 10.0,
            "baseline_r2": 0.1,
            "sce_rmse": 9.0,
            "sce_r2": 0.2,
            "rmse_improvement_pct": 10.0,
            "r2_improvement_pp": 0.1,
            "n_baseline_features": 3,
            "n_sce_features": 10,
        },
        n_rows_loaded=2,
        n_rows_after_filter=2,
    )

    for key in [
        "run_id",
        "run_grade",
        "timestamp_utc",
        "git_sha",
        "git_dirty",
        "config_path",
        "config_hash",
        "dataset",
        "split",
        "sce",
        "model",
        "metrics",
        "diagnostics",
        "promotion",
    ]:
        assert key in metadata


def test_run_grade_dirty_repo_blocks_report_grade(monkeypatch):
    monkeypatch.setattr("scripts.run._git_dirty", lambda: True)
    with pytest.raises(RuntimeError, match="clean git tree"):
        _enforce_run_grade_policy("report-grade", allow_dirty=False)


def test_report_grade_blocks_when_diagnostics_missing(tmp_path: Path):
    cfg = tmp_path / "sample.toml"
    cfg.write_text("[dataset]\npath='data.parquet'\n[target]\ncolumn='price'\n", encoding="utf-8")

    df = pd.DataFrame({"price": [1.0, 2.0]})
    split = DatasetSplit(
        train_idx=pd.Index([0]),
        test_idx=pd.Index([1]),
        train_df=df.iloc[[0]],
        test_df=df.iloc[[1]],
    )
    sce_config = ContextConfig(
        target_col="price",
        categorical_cols=["city"],
        aggregations=[AggregationMethod.MEAN],
        use_cross_fitting=False,
    )

    with pytest.raises(RuntimeError, match="missing_diagnostic"):
        _collect_run_metadata(
            config_name="sample",
            config_path=cfg,
            config={
                "dataset": {"source": "local", "path": "data/parquet/sample.parquet"},
                "target": {"column": "price"},
                "split": {"strategy": "random", "test_size": 0.5, "random_state": 42},
                "sce": {},
            },
            run_grade="report-grade",
            split=split,
            sce_config=sce_config,
            model_type="xgboost",
            model_params={"n_estimators": 10},
            runtime_seconds=1.5,
            metrics={"baseline_rmse": 10.0, "sce_rmse": 9.0},
            n_rows_loaded=2,
            n_rows_after_filter=2,
            diagnostics={
                "permuted_target": None,
                "shuffled_groups": None,
                "crossfit_ab": None,
                "feature_dominance": None,
            },
        )


def test_report_grade_blocks_for_dominant_target_global_feature(tmp_path: Path):
    cfg = tmp_path / "sample.toml"
    cfg.write_text("[dataset]\npath='data.parquet'\n[target]\ncolumn='price'\n", encoding="utf-8")

    df = pd.DataFrame({"price": [1.0, 2.0]})
    split = DatasetSplit(
        train_idx=pd.Index([0]),
        test_idx=pd.Index([1]),
        train_df=df.iloc[[0]],
        test_df=df.iloc[[1]],
    )
    sce_config = ContextConfig(
        target_col="price",
        categorical_cols=["city"],
        aggregations=[AggregationMethod.MEAN],
        use_cross_fitting=False,
    )

    with pytest.raises(RuntimeError, match="feature_dominance:target_global"):
        _collect_run_metadata(
            config_name="sample",
            config_path=cfg,
            config={
                "dataset": {"source": "local", "path": "data/parquet/sample.parquet"},
                "target": {"column": "price"},
                "split": {"strategy": "random", "test_size": 0.5, "random_state": 42},
                "sce": {},
            },
            run_grade="report-grade",
            split=split,
            sce_config=sce_config,
            model_type="xgboost",
            model_params={"n_estimators": 10},
            runtime_seconds=1.5,
            metrics={"baseline_rmse": 10.0, "sce_rmse": 9.0},
            n_rows_loaded=2,
            n_rows_after_filter=2,
            diagnostics={
                "permuted_target": {"pass": True},
                "shuffled_groups": {"pass": True},
                "crossfit_ab": {"leakage_signal_pp": 2.0},
                "feature_dominance": {
                    "dominated": True,
                    "top_features": ["global_price_mean"],
                    "top_k_share_pct": 90.0,
                },
            },
        )


def test_report_grade_block_is_written_to_ledger(tmp_path: Path):
    cfg = tmp_path / "sample.toml"
    cfg.write_text("[dataset]\npath='data.parquet'\n[target]\ncolumn='price'\n", encoding="utf-8")

    df = pd.DataFrame({"price": [1.0, 2.0]})
    split = DatasetSplit(
        train_idx=pd.Index([0]),
        test_idx=pd.Index([1]),
        train_df=df.iloc[[0]],
        test_df=df.iloc[[1]],
    )
    sce_config = ContextConfig(
        target_col="price",
        categorical_cols=["city"],
        aggregations=[AggregationMethod.MEAN],
        use_cross_fitting=False,
    )
    block_file = tmp_path / "report_grade_blocks.jsonl"

    with pytest.raises(RuntimeError, match="missing_diagnostic"):
        _collect_run_metadata(
            config_name="sample",
            config_path=cfg,
            config={
                "dataset": {"source": "local", "path": "data/parquet/sample.parquet"},
                "target": {"column": "price"},
                "split": {"strategy": "random", "test_size": 0.5, "random_state": 42},
                "sce": {},
            },
            run_grade="report-grade",
            split=split,
            sce_config=sce_config,
            model_type="xgboost",
            model_params={"n_estimators": 10},
            runtime_seconds=1.5,
            metrics={"baseline_rmse": 10.0, "sce_rmse": 9.0},
            n_rows_loaded=2,
            n_rows_after_filter=2,
            diagnostics={
                "permuted_target": None,
                "shuffled_groups": None,
                "crossfit_ab": None,
                "feature_dominance": None,
            },
            report_grade_block_file=block_file,
        )

    lines = block_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["dataset"] == "sample"
    assert "missing_diagnostic:permuted_target" in payload["blocked_by"]
