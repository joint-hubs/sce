"""
@module: tests.test_models
@depends: sce.models
@exports:
@data_flow: model params -> estimator construction
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest

from sce.models import build_model, extract_feature_importance, model_supports_gpu

HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
HAS_CATBOOST = importlib.util.find_spec("catboost") is not None

# sklearn estimators that reject raw NaN input and are therefore wrapped in an
# imputing pipeline by build_model.
NAN_INTOLERANT_MODELS = ["ridge", "random_forest", "extra_trees", "gradient_boosting"]


def _nan_regression_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(size=40),
            "b": rng.normal(size=40),
            "c": rng.normal(size=40),
        }
    )
    # Inject missing values the way SCE feature blocks do for small/empty groups.
    X.loc[0:5, "a"] = np.nan
    X.loc[10:12, "b"] = np.nan
    y = pd.Series(2.0 * X["c"].fillna(0.0) + rng.normal(scale=0.1, size=40))
    return X, y


@pytest.mark.parametrize("model_type", NAN_INTOLERANT_MODELS)
def test_build_model_handles_nan_inputs(model_type):
    """sklearn models must fit and predict on NaN-containing features."""
    X, y = _nan_regression_data()
    model = build_model(model_type)
    model.fit(X, y)
    preds = model.predict(X)
    assert np.isfinite(preds).all()

    importance = extract_feature_importance(model, X.columns)
    assert set(importance["feature"]) == set(X.columns)
    assert importance["importance"].ge(0).all()


def test_model_supports_gpu_only_for_accelerated_backends():
    assert model_supports_gpu("xgboost") is True
    assert model_supports_gpu("lightgbm") is True
    assert model_supports_gpu("catboost") is True
    assert model_supports_gpu("ridge") is False


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm not installed")
def test_build_model_passes_lightgbm_gpu_flags():
    model = build_model("lightgbm", {"use_gpu": True, "gpu_device_id": 0})

    params = model.get_params()
    assert params["device"] == "gpu"
    assert params["gpu_device_id"] == 0


@pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")
def test_build_model_passes_catboost_gpu_flags():
    model = build_model("catboost", {"use_gpu": True, "gpu_device_id": 0})

    params = model.get_params()
    assert params["task_type"] == "GPU"
    assert params["devices"] == "0"


@pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")
def test_build_model_normalizes_catboost_aliases():
    model = build_model(
        "catboost",
        {
            "iterations": 300,
            "n_estimators": 100,
            "depth": 6,
            "max_depth": 4,
            "random_state": 7,
            "colsample_bytree": 0.75,
        },
    )

    params = model.get_params()
    assert params["iterations"] == 300
    assert params["depth"] == 6
    assert params["random_seed"] == 7
    assert params["rsm"] == 0.75
    assert "n_estimators" not in params
    assert "max_depth" not in params
