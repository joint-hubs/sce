"""
@module: tests.equity.test_walk_forward_monotonicity
@depends: equity.diagnostics.walk_forward_monotonicity
@exports:
@data_flow: synthetic fold lists -> run_walk_forward_monotonicity -> pass/fail

S4.5 walk-forward monotonicity diagnostic unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from equity.diagnostics.walk_forward_monotonicity import (
    normalize_folds,
    run_walk_forward_monotonicity,
)


def test_clean_folds_pass() -> None:
    folds = [
        {
            "train_max": "2020-01-01",
            "val_min": "2020-01-02",
            "val_max": "2020-03-01",
        },
        {
            "train_max": "2020-03-01",
            "val_min": "2020-03-02",
            "val_max": "2020-06-01",
            "test_min": "2020-06-02",
            "test_max": "2020-09-01",
        },
    ]
    result = run_walk_forward_monotonicity(folds)
    assert result["pass"] is True
    assert result["n_violations"] == 0
    assert result["n_folds"] == 2
    assert result["violations"] == []
    assert result["strict"] is False


def test_leaky_fold_fails() -> None:
    folds = [
        {
            "train_max": "2020-02-01",
            "val_min": "2020-01-15",  # leak: train ends after val starts
            "val_max": "2020-03-01",
        }
    ]
    result = run_walk_forward_monotonicity(folds)
    assert result["pass"] is False
    assert result["n_violations"] == 1
    v = result["violations"][0]
    assert v["fold_idx"] == 0
    assert "train_max" in v["reason"] or "val_min" in v["reason"]
    assert "train_max" in v
    assert "val_min" in v


def test_gap_zero_allowed_by_default() -> None:
    folds = [
        {
            "train_max": "2020-01-01",
            "val_min": "2020-01-01",  # gap=0 — PRD §8.1 allows
            "val_max": "2020-03-01",
        }
    ]
    result = run_walk_forward_monotonicity(folds, strict=False)
    assert result["pass"] is True
    assert result["n_violations"] == 0


def test_strict_makes_equality_a_violation() -> None:
    folds = [
        {
            "train_max": "2020-01-01",
            "val_min": "2020-01-01",
            "val_max": "2020-03-01",
        }
    ]
    result = run_walk_forward_monotonicity(folds, strict=True)
    assert result["pass"] is False
    assert result["n_violations"] == 1
    assert result["strict"] is True
    assert result["violations"][0]["reason"] == "train_max_not_before_val_min"


def test_test_bound_violation() -> None:
    folds = [
        {
            "train_max": "2020-01-01",
            "val_min": "2020-01-02",
            "val_max": "2020-06-01",
            "test_min": "2020-05-01",  # leak into test
            "test_max": "2020-09-01",
        }
    ]
    result = run_walk_forward_monotonicity(folds)
    assert result["pass"] is False
    reasons = {v["reason"] for v in result["violations"]}
    assert "val_max_after_test_min" in reasons


def test_accepts_sce_last_fold_timestamps_shape() -> None:
    """Adapter: SCE engine._last_fold_timestamps shape works unchanged."""
    sce_folds = [
        {
            "train_size": 80,
            "val_size": 20,
            "train_max": pd.Timestamp("2024-03-01", tz="UTC"),
            "val_min": pd.Timestamp("2024-03-02", tz="UTC"),
            "val_max": pd.Timestamp("2024-04-01", tz="UTC"),
        },
        {
            "train_size": 100,
            "val_size": 20,
            "train_max": pd.Timestamp("2024-04-01", tz="UTC"),
            "val_min": pd.Timestamp("2024-04-02", tz="UTC"),
            "val_max": pd.Timestamp("2024-05-01", tz="UTC"),
        },
    ]
    normalized = normalize_folds(sce_folds)
    assert len(normalized) == 2
    assert "train_size" in normalized[0]  # passthrough extra keys

    result = run_walk_forward_monotonicity(sce_folds)
    assert result["pass"] is True
    assert result["n_folds"] == 2
    assert result["n_violations"] == 0


def test_missing_required_keys_raises() -> None:
    with pytest.raises(ValueError, match="missing required keys"):
        run_walk_forward_monotonicity([{"train_max": "2020-01-01"}])


def test_between_fold_overlap_fails() -> None:
    """Overlapping val windows across folds must fail between-fold check."""
    folds = [
        {
            "train_max": "2020-01-01",
            "val_min": "2020-01-02",
            "val_max": "2020-04-01",  # overlaps next val
        },
        {
            "train_max": "2020-03-01",
            "val_min": "2020-03-02",  # starts before prev val_max
            "val_max": "2020-06-01",
        },
    ]
    result = run_walk_forward_monotonicity(folds)
    assert result["pass"] is False
    reasons = {v["reason"] for v in result["violations"]}
    assert "val_max_after_next_val_min" in reasons


def test_between_fold_reverse_order_fails() -> None:
    """Reverse-ordered val windows must fail between-fold progression."""
    folds = [
        {
            "train_max": "2020-06-01",
            "val_min": "2020-06-02",
            "val_max": "2020-09-01",
        },
        {
            "train_max": "2020-01-01",
            "val_min": "2020-01-02",
            "val_max": "2020-03-01",
        },
    ]
    result = run_walk_forward_monotonicity(folds)
    assert result["pass"] is False
    reasons = {v["reason"] for v in result["violations"]}
    assert "val_max_after_next_val_min" in reasons


def test_between_fold_forward_progression_passes() -> None:
    """Non-overlapping forward-progressing vals pass (including gap=0)."""
    folds = [
        {
            "train_max": "2020-01-01",
            "val_min": "2020-01-02",
            "val_max": "2020-03-01",
        },
        {
            "train_max": "2020-03-01",
            "val_min": "2020-03-01",  # gap=0 between folds allowed non-strict
            "val_max": "2020-06-01",
        },
        {
            "train_max": "2020-06-01",
            "val_min": "2020-06-02",
            "val_max": "2020-09-01",
        },
    ]
    result = run_walk_forward_monotonicity(folds)
    assert result["pass"] is True
    assert result["n_violations"] == 0


def test_cli_inline_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from equity.diagnostics import walk_forward_monotonicity as mod

    # Point RESULTS_DIR into tmp so the CLI does not write into the repo.
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)

    folds_json = json.dumps(
        [{"train_max": "2020-01-01", "val_min": "2020-01-02", "val_max": "2020-03-01"}]
    )
    monkeypatch.setattr(
        "sys.argv",
        ["walk_forward_monotonicity", "--folds", folds_json],
    )
    rc = mod.main()
    assert rc == 0
