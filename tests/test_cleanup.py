"""
Tests for feature cleanup pipeline.
"""

import numpy as np
import pandas as pd

from sce.cleanup import FeatureCleanupPipeline
from sce.config import CleanupConfig


def test_cleanup_removes_constant_and_leakage():
    rng = np.random.default_rng(42)
    y = pd.Series(rng.normal(size=100))
    X = pd.DataFrame({
        "constant": 1.0,
        "leaky": y * 1.0,
        "noise": rng.normal(size=100),
    })

    config = CleanupConfig(
        leakage_remove_threshold=0.9,
        leakage_warn_threshold=0.8,
        correlation_enabled=False,
        hierarchy_enabled=False,
    )
    pipeline = FeatureCleanupPipeline(config)
    cleaned, report = pipeline.fit_transform(X, y, target_col="price")

    assert "constant" not in cleaned.columns
    assert "leaky" not in cleaned.columns
    assert report.total_removed >= 2


def test_cleanup_removes_correlated_pair():
    rng = np.random.default_rng(123)
    base = rng.normal(size=200)
    X = pd.DataFrame({
        "feat_a": base,
        "feat_b": base * 0.99 + rng.normal(scale=0.01, size=200),
    })
    y = pd.Series(base + rng.normal(scale=0.5, size=200))

    config = CleanupConfig(
        leakage_enabled=False,
        correlation_threshold=0.95,
        correlation_drop_strategy="lower_target_corr",
        hierarchy_enabled=False,
    )
    pipeline = FeatureCleanupPipeline(config)
    cleaned, report = pipeline.fit_transform(X, y, target_col="price")

    assert len(cleaned.columns) == 1
    assert report.correlation_removed


def test_cleanup_hierarchy_redundancy_prefers_child():
    rng = np.random.default_rng(7)
    base = rng.normal(size=120)
    X = pd.DataFrame({
        "city_price_mean": base,
        "city__neighborhood_price_mean": base + rng.normal(scale=0.001, size=120),
    })
    y = pd.Series(base + rng.normal(scale=0.1, size=120))

    config = CleanupConfig(
        leakage_enabled=False,
        correlation_enabled=False,
        hierarchy_enabled=True,
        hierarchy_prefer="child",
        hierarchy_corr_threshold=0.9,
    )
    pipeline = FeatureCleanupPipeline(config)
    cleaned, report = pipeline.fit_transform(X, y, target_col="price")

    assert "city_price_mean" not in cleaned.columns
    assert "city__neighborhood_price_mean" in cleaned.columns
    assert report.hierarchy_removed
