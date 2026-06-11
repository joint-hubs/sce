"""Tests for model preset resolution."""

from sce.model_presets import resolve_model_presets, resolve_xgboost_presets


def test_resolve_xgboost_presets_defaults():
    run_cfg = {}
    model_cfg = {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.2}
    names, presets = resolve_xgboost_presets(run_cfg, model_cfg)
    assert "default" in names
    assert "default" in presets
    assert presets["default"]["n_estimators"] == 50


def test_resolve_model_presets_uses_model_specific_run_key():
    run_cfg = {"ridge_configs": ["strong", "default"]}
    model_cfg = {"type": "ridge", "alpha": 2.5}

    model_type, names, presets = resolve_model_presets(run_cfg, model_cfg)

    assert model_type == "ridge"
    assert names == ["strong", "default"]
    assert presets["default"]["alpha"] == 2.5
    assert presets["strong"]["alpha"] == 2.5


def test_resolve_model_presets_merges_legacy_model_overrides():
    run_cfg = {}
    model_cfg = {"type": "random_forest", "n_estimators": 50, "max_depth": 5}

    model_type, names, presets = resolve_model_presets(run_cfg, model_cfg)

    assert model_type == "random_forest"
    assert names == ["default"]
    assert presets["default"]["n_estimators"] == 50
    assert presets["default"]["max_depth"] == 5


def test_resolve_model_presets_applies_model_overrides_to_named_presets():
    run_cfg = {"lightgbm_configs": ["default", "shallow"]}
    model_cfg = {"type": "lightgbm", "use_gpu": True}

    model_type, names, presets = resolve_model_presets(run_cfg, model_cfg)

    assert model_type == "lightgbm"
    assert names == ["default", "shallow"]
    assert presets["default"]["use_gpu"] is True
    assert presets["shallow"]["use_gpu"] is True
