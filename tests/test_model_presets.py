"""Tests for XGBoost preset resolution."""

from sce.model_presets import resolve_xgboost_presets


def test_resolve_xgboost_presets_defaults():
    run_cfg = {}
    model_cfg = {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.2}
    names, presets = resolve_xgboost_presets(run_cfg, model_cfg)
    assert "default" in names
    assert "default" in presets
    assert presets["default"]["n_estimators"] == 50
