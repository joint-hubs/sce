"""Regression tests for the experiment runner protocol."""

import pandas as pd
import pytest

from scripts.run import (
    _build_sce_config,
    _resolve_categorical_mode,
    _run_sce_enrichment,
    _split_dataset,
    prepare_features,
)


def test_build_sce_config_uses_manual_or_auto_categorical_mode():
    config = {
        "features": {"categorical": ["city"]},
        "sce": {
            "aggregations": ["mean"],
            "use_cross_fitting": False,
            "include_interactions": False,
        },
    }

    manual_cfg = _build_sce_config(config, "price", categorical_mode="manual")
    auto_cfg = _build_sce_config(config, "price", categorical_mode="auto")

    assert manual_cfg.categorical_cols == ["city"]
    assert auto_cfg.categorical_cols is None


def test_resolve_categorical_mode_rejects_invalid_value():
    with pytest.raises(ValueError, match="categorical_mode"):
        _resolve_categorical_mode({"run": {"categorical_mode": "invalid"}})


def test_runner_enriches_test_split_with_train_only_statistics():
    """Test rows should be transformed with train-derived statistics only."""
    train_df = pd.DataFrame(
        {
            "city": ["A", "A", "B", "B"],
            "sqft": [10, 20, 30, 40],
            "price": [10.0, 20.0, 100.0, 110.0],
        },
        index=[0, 1, 2, 3],
    )
    test_df = pd.DataFrame(
        {
            "city": ["A", "B"],
            "sqft": [25, 45],
            "price": [30.0, 120.0],
        },
        index=[4, 5],
    )
    config = {
        "features": {
            "numeric": ["sqft"],
            "categorical": ["city"],
        },
        "sce": {
            "aggregations": ["mean"],
            "min_group_size": 1,
            "use_cross_fitting": False,
            "include_interactions": False,
            "include_fold_variance": False,
            "max_cardinality": 10,
        },
    }

    enrichment = _run_sce_enrichment(
        train_df,
        config,
        "price",
        transform_df=test_df,
    )
    enriched = enrichment.df_enriched

    assert enriched.loc[4, "city_price_mean"] == pytest.approx(15.0)
    assert enriched.loc[5, "city_price_mean"] == pytest.approx(105.0)
    assert enriched.loc[4, "city_price_mean"] != pytest.approx(20.0)
    assert enriched.loc[5, "city_price_mean"] != pytest.approx(110.0)
    assert enriched.loc[4, "price"] == pytest.approx(30.0)
    assert enriched.loc[5, "price"] == pytest.approx(120.0)


def test_runner_target_mean_variant_uses_mean_only_features():
    train_df = pd.DataFrame(
        {
            "city": ["A", "A", "B", "B"],
            "room_type": ["x", "y", "x", "y"],
            "price": [10.0, 20.0, 100.0, 110.0],
        }
    )
    config = {
        "features": {"categorical": ["city", "room_type"]},
        "sce": {
            "aggregations": ["mean", "std", "count"],
            "min_group_size": 1,
            "use_cross_fitting": False,
            "include_interactions": True,
            "include_fold_variance": True,
        },
    }

    enrichment = _run_sce_enrichment(
        train_df,
        config,
        "price",
        context_variant="target_mean",
    )

    assert "city_price_mean" in enrichment.df_enriched.columns
    assert "city__room_type_price_mean" in enrichment.df_enriched.columns
    assert not any(col.endswith("_count") for col in enrichment.all_sce_cols)
    assert not any(col.endswith("_std") for col in enrichment.all_sce_cols)
    assert not any("fold_" in col for col in enrichment.all_sce_cols)


def test_runner_auto_mode_detects_categorical_columns():
    train_df = pd.DataFrame(
        {
            "city": ["A", "A", "B", "B"],
            "sqft": [10, 20, 30, 40],
            "price": [10.0, 20.0, 100.0, 110.0],
        }
    )
    config = {
        "features": {"numeric": ["sqft"]},
        "sce": {
            "aggregations": ["mean"],
            "min_group_size": 1,
            "use_cross_fitting": False,
            "include_interactions": False,
            "include_fold_variance": False,
            "max_cardinality": 10,
        },
    }

    enrichment = _run_sce_enrichment(
        train_df,
        config,
        "price",
        categorical_mode="auto",
    )

    assert "city_price_mean" in enrichment.df_enriched.columns


def test_runner_mean_count_variant_adds_count_without_std():
    train_df = pd.DataFrame(
        {
            "city": ["A", "A", "B", "B"],
            "price": [10.0, 20.0, 100.0, 110.0],
        }
    )
    config = {
        "features": {"categorical": ["city"]},
        "sce": {
            "aggregations": ["mean", "std", "count"],
            "min_group_size": 1,
            "use_cross_fitting": False,
            "include_interactions": False,
        },
    }

    enrichment = _run_sce_enrichment(
        train_df,
        config,
        "price",
        context_variant="hierarchical_mean_count",
    )

    assert "city_price_mean" in enrichment.df_enriched.columns
    assert "city_price_count" in enrichment.df_enriched.columns
    assert "city_price_std" not in enrichment.df_enriched.columns


def test_temporal_split_holds_out_latest_periods():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-04",
                ]
            ),
            "series": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "price": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    config = {"split": {"strategy": "temporal", "time_col": "date", "test_periods": 2}}

    split = _split_dataset(df, "price", config=config)

    assert split.train_df["date"].max() < split.test_df["date"].min()
    assert set(split.test_df["date"].dt.strftime("%Y-%m-%d")) == {"2024-01-03", "2024-01-04"}


def test_categorical_encoding_train_test_consistent():
    config = {
        "features": {
            "categorical": ["cat"],
            "numeric": ["num"],
        }
    }
    train_df = pd.DataFrame({"cat": ["a", "b", "c"], "num": [1, 2, 3], "price": [10, 20, 30]})
    test_df = pd.DataFrame({"cat": ["c", "a", "b"], "num": [4, 5, 6], "price": [40, 50, 60]})

    train_prepared = prepare_features(train_df, config, "price")
    test_prepared = prepare_features(
        test_df,
        config,
        "price",
        encoder=train_prepared.encoder,
        droplist=train_prepared.droplist,
    )

    train_codes = dict(zip(train_df["cat"], train_prepared.X["cat"].tolist()))
    test_codes = dict(zip(test_df["cat"], test_prepared.X["cat"].tolist()))
    assert train_codes["a"] == test_codes["a"]
    assert train_codes["b"] == test_codes["b"]
    assert train_codes["c"] == test_codes["c"]


def test_unseen_category_in_test():
    config = {
        "features": {
            "categorical": ["cat"],
            "numeric": ["num"],
        }
    }
    train_df = pd.DataFrame({"cat": ["a", "b"], "num": [1, 2], "price": [10, 20]})
    test_df = pd.DataFrame({"cat": ["z"], "num": [3], "price": [30]})

    train_prepared = prepare_features(train_df, config, "price")
    test_prepared = prepare_features(
        test_df,
        config,
        "price",
        encoder=train_prepared.encoder,
        droplist=train_prepared.droplist,
    )

    assert pd.isna(test_prepared.X.iloc[0]["cat"])
    assert test_prepared.unseen_categorical["cat"]["unseen_count"] == 1
    assert test_prepared.unseen_categorical["cat"]["samples"] == ["z"]


def test_pruning_droplist_train_only():
    config = {
        "features": {
            "numeric": ["x", "z"],
            "categorical": [],
        },
        "run": {
            "feature_pruning": {
                "missing_threshold": 0.4,
                "drop_zero_variance": False,
            }
        },
    }
    train_df = pd.DataFrame(
        {"x": [1.0, None, None, 4.0], "z": [10, 11, 12, 13], "price": [1, 2, 3, 4]}
    )
    test_df = pd.DataFrame(
        {"x": [10.0, 11.0, 12.0, 13.0], "z": [20, 21, 22, 23], "price": [5, 6, 7, 8]}
    )

    train_prepared = prepare_features(train_df, config, "price")
    test_prepared = prepare_features(
        test_df,
        config,
        "price",
        encoder=train_prepared.encoder,
        droplist=train_prepared.droplist,
    )

    assert "x" not in train_prepared.X.columns
    assert "x" not in test_prepared.X.columns
    assert "z" in train_prepared.X.columns


def test_pruning_zero_variance_train_only():
    config = {
        "features": {
            "numeric": ["x", "z"],
            "categorical": [],
        },
        "run": {
            "feature_pruning": {
                "missing_threshold": 1.0,
                "drop_zero_variance": True,
            }
        },
    }
    train_df = pd.DataFrame(
        {"x": [1.0, 1.0, 1.0, 1.0], "z": [2.0, 3.0, 4.0, 5.0], "price": [1, 2, 3, 4]}
    )
    test_df = pd.DataFrame(
        {"x": [10.0, 11.0, 12.0, 13.0], "z": [6.0, 7.0, 8.0, 9.0], "price": [5, 6, 7, 8]}
    )

    train_prepared = prepare_features(train_df, config, "price")
    test_prepared = prepare_features(
        test_df,
        config,
        "price",
        encoder=train_prepared.encoder,
        droplist=train_prepared.droplist,
    )

    assert "x" not in train_prepared.X.columns
    assert "x" not in test_prepared.X.columns
    assert "z" in train_prepared.X.columns
