"""Tests for scikit-learn pipeline integration."""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sce import ContextConfig, create_sce_pipeline, fit_context_pipeline


def test_create_sce_pipeline_returns_pipeline(basic_config):
    """The public helper should create a sklearn pipeline."""
    pipeline = create_sce_pipeline(basic_config)

    assert [name for name, _ in pipeline.steps] == ["sce"]


def test_pipeline_with_model_accepts_fit_x_y(sample_data):
    """The sklearn pipeline path should work with fit(X, y)."""
    encoded = sample_data.copy()
    encoded["city"] = pd.Categorical(encoded["city"]).codes
    encoded["neighborhood"] = pd.Categorical(encoded["neighborhood"]).codes

    config = ContextConfig(
        categorical_cols=["city", "neighborhood"],
        target_col="price",
        use_cross_fitting=True,
        n_folds=3,
    )
    pipeline = create_sce_pipeline(
        config, model=RandomForestRegressor(n_estimators=10, random_state=42)
    )

    X = encoded.drop(columns=["price"])
    y = encoded["price"]
    pipeline.fit(X, y)
    predictions = pipeline.predict(X.head(5))

    assert len(predictions) == 5


def test_fit_context_pipeline_handles_feature_only_input(sample_data):
    """The convenience wrapper should fit using the target column from the frame."""
    encoded = sample_data.copy()
    encoded["city"] = pd.Categorical(encoded["city"]).codes
    encoded["neighborhood"] = pd.Categorical(encoded["neighborhood"]).codes

    config = ContextConfig(
        categorical_cols=["city", "neighborhood"],
        target_col="price",
        use_cross_fitting=True,
        n_folds=3,
    )
    pipeline = fit_context_pipeline(
        encoded,
        config,
        model=RandomForestRegressor(n_estimators=10, random_state=42),
    )

    predictions = pipeline.predict(encoded.drop(columns=["price"]).head(3))

    assert len(predictions) == 3
