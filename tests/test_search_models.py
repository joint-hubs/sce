"""
@module: tests.test_search_models
@depends: sce.search
@exports:
@data_flow: toy regression data -> train_model -> metrics + importance
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from sce.search import train_model


HAS_LIGHTGBM = importlib.util.find_spec("lightgbm") is not None
HAS_CATBOOST = importlib.util.find_spec("catboost") is not None


def _toy_regression_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    X_train = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_b": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
        }
    )
    y_train = pd.Series([1.0, 2.2, 3.9, 6.1, 7.8, 10.2])
    X_test = pd.DataFrame(
        {
            "feature_a": [1.5, 2.5, 4.5],
            "feature_b": [1.75, 2.25, 3.25],
        }
    )
    y_test = pd.Series([2.9, 5.1, 8.9])
    return X_train, y_train, X_test, y_test


def test_train_model_supports_ridge():
    X_train, y_train, X_test, y_test = _toy_regression_data()

    _, metrics, importance = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_type="ridge",
        config_name="default",
    )

    assert set(metrics) == {"rmse", "r2", "mae"}
    assert list(importance["feature"]) == ["feature_a", "feature_b"]
    assert importance["importance"].ge(0).all()


def test_train_model_supports_random_forest():
    X_train, y_train, X_test, y_test = _toy_regression_data()

    _, metrics, importance = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_type="random_forest",
        config_name="shallow",
    )

    assert set(metrics) == {"rmse", "r2", "mae"}
    assert set(importance["feature"]) == {"feature_a", "feature_b"}
    assert importance["importance"].ge(0).all()


@pytest.mark.skipif(not HAS_LIGHTGBM, reason="lightgbm not installed")
def test_train_model_supports_lightgbm():
    X_train, y_train, X_test, y_test = _toy_regression_data()

    _, metrics, importance = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_type="lightgbm",
        config_name="shallow",
    )

    assert set(metrics) == {"rmse", "r2", "mae"}
    assert set(importance["feature"]) == {"feature_a", "feature_b"}
    assert importance["importance"].ge(0).all()


@pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not installed")
def test_train_model_supports_catboost():
    X_train, y_train, X_test, y_test = _toy_regression_data()

    _, metrics, importance = train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        model_type="catboost",
        config_name="shallow",
    )

    assert set(metrics) == {"rmse", "r2", "mae"}
    assert set(importance["feature"]) == {"feature_a", "feature_b"}
    assert importance["importance"].ge(0).all()