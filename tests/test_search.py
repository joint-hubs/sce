"""Regression tests for search experiment robustness."""

import pandas as pd

from sce.importance import run_iterative_pruning
from sce.search import FeatureCombinationSearch


def test_search_ignores_features_missing_from_test_split():
    """Search should skip train-only features instead of failing on column lookup."""
    X_train = pd.DataFrame(
        {
            "stable_base": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "train_only_base": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "stable_context": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "train_only_context": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0],
        }
    )
    y_train = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

    X_test = pd.DataFrame(
        {
            "stable_base": [1.5, 2.5, 3.5],
            "stable_context": [0.25, 0.35, 0.45],
        }
    )
    y_test = pd.Series([11.0, 13.0, 15.0])

    searcher = FeatureCombinationSearch(
        base_features=["stable_base", "train_only_base"],
        context_features=["stable_context", "train_only_context"],
        sampling_pct=100.0,
        min_samples=4,
        max_samples=4,
        model_configs=["default"],
        model_type="ridge",
        run_ablation=False,
        run_significance_selection=False,
        random_state=0,
    )

    summary = searcher.search(X_train, y_train, X_test, y_test)

    assert summary.all_results
    assert all("train_only_base" not in result.features for result in summary.all_results)
    assert all("train_only_context" not in result.features for result in summary.all_results)
    assert any(result.strategy == "baseline" for result in summary.all_results)


def test_iterative_pruning_ignores_features_missing_from_test_split():
    """Pruning should use only columns shared by train and test matrices."""
    X_train = pd.DataFrame(
        {
            "stable_base": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "train_only_base": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "stable_context": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        }
    )
    y_train = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])

    X_test = pd.DataFrame(
        {
            "stable_base": [1.5, 2.5, 3.5],
            "stable_context": [0.25, 0.35, 0.45],
        }
    )
    y_test = pd.Series([11.0, 13.0, 15.0])

    pruning_results, removed_df = run_iterative_pruning(
        X_train,
        y_train,
        X_test,
        y_test,
        features=["stable_base", "train_only_base", "stable_context"],
        model_type="ridge",
    )

    assert pruning_results
    assert all("train_only_base" not in result.features for result in pruning_results)
    if not removed_df.empty:
        assert "train_only_base" not in removed_df["feature"].tolist()