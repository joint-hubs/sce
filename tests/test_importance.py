"""Tests for importance aggregation."""

import pandas as pd

from sce.importance import aggregate_importance
from sce.search import SearchResult


def test_aggregate_importance_basic():
    importance = pd.DataFrame({
        "feature": ["a", "b"],
        "importance": [0.7, 0.3],
    })
    result = SearchResult(
        config_id=0,
        strategy="baseline",
        n_features=2,
        n_base=2,
        n_context=0,
        rmse=1.0,
        r2=0.5,
        mae=0.8,
        features=["a", "b"],
        model_config="default",
        feature_importance=importance,
    )

    aggregated = aggregate_importance([result])
    assert not aggregated.empty
    assert "feature" in aggregated.columns
    assert "importance_mean" in aggregated.columns
