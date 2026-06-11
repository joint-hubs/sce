"""Tests for failure-case logging ledger."""

import json
from pathlib import Path

from scripts.run import _record_failure_case


def test_record_failure_case_writes_jsonl(tmp_path: Path):
    out_file = tmp_path / "failure_cases.jsonl"

    _record_failure_case(
        run_id="demo_run",
        config_name="demo_dataset",
        model_type="xgboost",
        rmse_improvement_pct=0.5,
        runtime_seconds=12.0,
        baseline_runtime_seconds=5.0,
        results_file=out_file,
    )

    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["run_id"] == "demo_run"
    assert "rmse_improvement_lt_1pct" in payload["reason"]
    assert "runtime_gt_2x_baseline" in payload["reason"]


def test_record_failure_case_skips_healthy_runs(tmp_path: Path):
    out_file = tmp_path / "failure_cases.jsonl"

    _record_failure_case(
        run_id="healthy_run",
        config_name="demo_dataset",
        model_type="xgboost",
        rmse_improvement_pct=3.0,
        runtime_seconds=8.0,
        baseline_runtime_seconds=5.0,
        results_file=out_file,
    )

    assert not out_file.exists()
