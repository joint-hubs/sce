"""Tests for diagnostics runners."""

from pathlib import Path

import pandas as pd

from scripts.diagnostics import crossfit_ab, feature_dominance, permuted_target, shuffled_groups


def _mock_loader(_config_name: str):
    config = {
        "target": {"column": "price"},
        "features": {"categorical": ["city"], "numeric": ["sqft"]},
        "sce": {"use_cross_fitting": True},
        "split": {"strategy": "random", "test_size": 0.2, "random_state": 42},
        "model": {"type": "xgboost", "n_estimators": 10, "max_depth": 2},
    }
    df = pd.DataFrame(
        {
            "city": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "sqft": [1, 2, 3, 4, 5, 6, 7, 8],
            "price": [10.0, 11.0, 20.0, 21.0, 30.0, 29.0, 40.0, 39.0],
        }
    )
    return config, Path("config.toml"), df, "price"


def _mock_eval(*_args, **_kwargs):
    return {"baseline_rmse": 10.0, "sce_rmse": 9.0}


def test_permuted_target_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(permuted_target, "load_config_and_dataset", _mock_loader)
    monkeypatch.setattr(permuted_target, "evaluate_config_dataframe", _mock_eval)
    monkeypatch.setattr(permuted_target, "RESULTS_DIR", tmp_path)

    result = permuted_target.run_permuted_target("demo", n_permutations=2, seed=1, max_rows=None)
    assert "sce_advantage_permuted_mean" in result
    assert "pass" in result


def test_shuffled_groups_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(shuffled_groups, "load_config_and_dataset", _mock_loader)
    monkeypatch.setattr(shuffled_groups, "evaluate_config_dataframe", _mock_eval)
    monkeypatch.setattr(shuffled_groups, "RESULTS_DIR", tmp_path)

    result = shuffled_groups.run_shuffled_groups("demo", n_permutations=2, seed=1)
    assert "sce_advantage_shuffled_mean" in result
    assert "pass" in result


def test_shuffled_groups_per_column_breakdown(tmp_path, monkeypatch):
    monkeypatch.setattr(shuffled_groups, "load_config_and_dataset", _mock_loader)
    monkeypatch.setattr(shuffled_groups, "evaluate_config_dataframe", _mock_eval)
    monkeypatch.setattr(shuffled_groups, "RESULTS_DIR", tmp_path)

    result = shuffled_groups.run_shuffled_groups("demo", n_permutations=2, seed=1, mode="per-column")
    assert result["mode"] == "per-column"
    assert "city" in result["columns_evaluated"]
    assert "city" in result["per_column"]
    assert len(result["per_column"]["city"]["advantages"]) == 2


def test_crossfit_ab_smoke(tmp_path, monkeypatch):
    monkeypatch.setattr(crossfit_ab, "load_config_and_dataset", _mock_loader)
    monkeypatch.setattr(crossfit_ab, "evaluate_config_dataframe", _mock_eval)
    monkeypatch.setattr(crossfit_ab, "RESULTS_DIR", tmp_path)

    result = crossfit_ab.run_crossfit_ab("demo")
    assert "rmse_cf" in result
    assert "leakage_signal_pp" in result


def test_permuted_target_report_grade_forces_full(monkeypatch):
    seen_sizes: list[int] = []

    def _eval(config, config_name, df, target_col, **kwargs):
        seen_sizes.append(len(df))
        return {"baseline_rmse": 10.0, "sce_rmse": 9.0}

    monkeypatch.setattr(permuted_target, "load_config_and_dataset", _mock_loader)
    monkeypatch.setattr(permuted_target, "evaluate_config_dataframe", _eval)

    result = permuted_target.run_permuted_target(
        "demo",
        n_permutations=1,
        seed=1,
        max_rows=3,
        run_grade="report-grade",
    )
    assert result["evaluated_rows"] == 8
    assert result["subsample_max_rows"] is None
    assert seen_sizes and all(size == 8 for size in seen_sizes)


def test_shuffled_groups_report_grade_forces_full(monkeypatch):
    seen_sizes: list[int] = []

    def _eval(config, config_name, df, target_col, **kwargs):
        seen_sizes.append(len(df))
        return {"baseline_rmse": 10.0, "sce_rmse": 9.0}

    monkeypatch.setattr(shuffled_groups, "load_config_and_dataset", _mock_loader)
    monkeypatch.setattr(shuffled_groups, "evaluate_config_dataframe", _eval)

    result = shuffled_groups.run_shuffled_groups(
        "demo",
        n_permutations=1,
        seed=1,
        max_rows=3,
        run_grade="report-grade",
    )
    assert result["evaluated_rows"] == 8
    assert result["subsample_max_rows"] is None
    assert seen_sizes and all(size == 8 for size in seen_sizes)


def test_crossfit_ab_report_grade_forces_full(monkeypatch):
    seen_sizes: list[int] = []

    def _eval(config, config_name, df, target_col, **kwargs):
        seen_sizes.append(len(df))
        return {"baseline_rmse": 10.0, "sce_rmse": 9.0, "sce_r2": 0.2}

    monkeypatch.setattr(crossfit_ab, "load_config_and_dataset", _mock_loader)
    monkeypatch.setattr(crossfit_ab, "evaluate_config_dataframe", _eval)

    result = crossfit_ab.run_crossfit_ab(
        "demo",
        max_rows=3,
        seed=1,
        run_grade="report-grade",
    )
    assert result["evaluated_rows"] == 8
    assert result["subsample_max_rows"] is None
    assert seen_sizes and all(size == 8 for size in seen_sizes)


def test_feature_dominance_synthetic(tmp_path):
    csv = tmp_path / "importance.csv"
    csv.write_text("feature,avg_importance\na,90\nb,5\nc,5\n", encoding="utf-8")

    result = feature_dominance.audit_feature_dominance_file(csv, top_k=3, threshold_pct=70)
    assert result["dominated"] is True


def test_feature_dominance_uniform(tmp_path):
    csv = tmp_path / "importance.csv"
    rows = [f"f{i},10" for i in range(10)]
    csv.write_text("feature,avg_importance\n" + "\n".join(rows) + "\n", encoding="utf-8")

    result = feature_dominance.audit_feature_dominance_file(csv, top_k=3, threshold_pct=70)
    assert result["dominated"] is False
