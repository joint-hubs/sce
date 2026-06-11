"""Leakage guard tests for temporal split and cross-fit strategies."""

import pandas as pd
import pytest

from sce import ContextConfig, StatisticalContextEngine
from sce.config import AggregationMethod
from scripts.run import _build_sce_config


def test_temporal_with_random_crossfit_raises():
    config = {
        "split": {"strategy": "temporal", "time_col": "date"},
        "sce": {"use_cross_fitting": True, "cross_fit_strategy": "random"},
        "features": {"categorical": ["city"]},
    }

    with pytest.raises(ValueError, match="Temporal split forbids random cross-fit"):
        _build_sce_config(config, target_col="price")


def test_temporal_with_rolling_crossfit_passes():
    config = {
        "split": {"strategy": "temporal", "time_col": "date"},
        "sce": {
            "use_cross_fitting": True,
            "cross_fit_strategy": "rolling",
            "aggregations": ["mean"],
            "include_interactions": False,
            "rolling_max_train_size": 10,
            "rolling_test_size": 5,
        },
        "features": {"categorical": ["city"]},
    }

    sce_config = _build_sce_config(config, target_col="price")
    assert sce_config.cross_fit_strategy == "rolling"


def test_rolling_crossfit_is_monotonic():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=40, freq="D"),
            "city": ["A"] * 20 + ["B"] * 20,
            "price": [float(i) for i in range(40)],
        }
    )

    config = ContextConfig(
        target_col="price",
        categorical_cols=["city"],
        aggregations=[AggregationMethod.MEAN],
        use_cross_fitting=True,
        cross_fit_strategy="rolling",
        time_col="date",
        n_folds=4,
        rolling_max_train_size=10,
        rolling_test_size=5,
        include_interactions=False,
    )

    engine = StatisticalContextEngine(config)
    enriched = engine.fit_transform(df)

    assert len(enriched) == len(df)
    assert engine._last_fold_timestamps
    for fold_info in engine._last_fold_timestamps:
        assert fold_info["train_max"] < fold_info["val_min"]
        assert fold_info["train_max"] < fold_info["val_max"]


def test_rolling_crossfit_train_window_is_bounded():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=60, freq="D"),
            "city": ["A"] * 30 + ["B"] * 30,
            "price": [float(i) for i in range(60)],
        }
    )

    config = ContextConfig(
        target_col="price",
        categorical_cols=["city"],
        aggregations=[AggregationMethod.MEAN],
        use_cross_fitting=True,
        cross_fit_strategy="rolling",
        time_col="date",
        n_folds=4,
        rolling_max_train_size=12,
        rolling_test_size=6,
        include_interactions=False,
    )

    engine = StatisticalContextEngine(config)
    engine.fit_transform(df)

    assert engine._last_fold_timestamps
    for fold_info in engine._last_fold_timestamps:
        assert fold_info["train_size"] <= 12
