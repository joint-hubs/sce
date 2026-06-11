"""Regression tests for search experiment robustness."""

import numpy as np
import pandas as pd

import sce.search as search_mod
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


def _make_search_data(n_train: int = 40, n_test: int = 7):
    rng = np.random.RandomState(0)
    X_train = pd.DataFrame(
        {
            "base_a": rng.normal(size=n_train),
            "ctx_a": rng.normal(size=n_train),
            "ctx_b": rng.normal(size=n_train),
        }
    )
    y_train = pd.Series(2 * X_train["base_a"] + X_train["ctx_a"] + rng.normal(scale=0.1, size=n_train))
    X_test = pd.DataFrame(
        {
            "base_a": rng.normal(size=n_test),
            "ctx_a": rng.normal(size=n_test),
            "ctx_b": rng.normal(size=n_test),
        }
    )
    y_test = pd.Series(2 * X_test["base_a"] + X_test["ctx_a"] + rng.normal(scale=0.1, size=n_test))
    return X_train, y_train, X_test, y_test


def test_search_selects_on_validation_and_touches_test_once(monkeypatch):
    """Candidates must be scored on the internal validation split; the test
    set may only be used for the final evaluation of the selected winners."""
    X_train, y_train, X_test, y_test = _make_search_data(n_train=40, n_test=7)

    eval_sizes = []
    real_train_model = search_mod.train_model

    def spy(X_tr, y_tr, X_ev, y_ev, *args, **kwargs):
        eval_sizes.append(len(X_ev))
        return real_train_model(X_tr, y_tr, X_ev, y_ev, *args, **kwargs)

    monkeypatch.setattr(search_mod, "train_model", spy)

    searcher = FeatureCombinationSearch(
        base_features=["base_a"],
        context_features=["ctx_a", "ctx_b"],
        sampling_pct=100.0,
        min_samples=4,
        max_samples=8,
        model_configs=["default"],
        model_type="ridge",
        run_ablation=False,
        run_significance_selection=False,
        random_state=0,
        val_fraction=0.25,
    )
    summary = searcher.search(X_train, y_train, X_test, y_test)

    n_val = len(searcher.val_indices_)
    assert n_val == 10  # 25% of 40 — distinct from the 7-row test set

    test_evals = [s for s in eval_sizes if s == len(X_test)]
    candidate_evals = [s for s in eval_sizes if s == n_val]
    # Test set evaluated at most once per distinct winner (rmse, r2, baseline)
    assert 1 <= len(test_evals) <= 3
    assert len(candidate_evals) == len(summary.all_results)
    assert len(test_evals) + len(candidate_evals) == len(eval_sizes)

    # Candidates carry validation metrics; winners carry test metrics
    assert all(r.eval_set == "validation" for r in summary.all_results)
    for winner in (summary.best_by_rmse, summary.best_by_r2, summary.baseline_result):
        assert winner.eval_set == "test"
        assert winner.val_rmse is not None

    # The winner must correspond to the best validation RMSE candidate
    best_candidate = min(summary.all_results, key=lambda r: r.rmse)
    assert summary.best_by_rmse.features == best_candidate.features
    assert summary.best_by_rmse.val_rmse == best_candidate.rmse


def test_search_tail_validation_holds_out_last_rows():
    """val_strategy='tail' must hold out the chronologically last train rows."""
    X_train, y_train, X_test, y_test = _make_search_data(n_train=20, n_test=5)

    searcher = FeatureCombinationSearch(
        base_features=["base_a"],
        context_features=["ctx_a"],
        sampling_pct=100.0,
        min_samples=2,
        max_samples=4,
        model_configs=["default"],
        model_type="ridge",
        run_ablation=False,
        run_significance_selection=False,
        random_state=0,
        val_fraction=0.2,
        val_strategy="tail",
    )
    searcher.search(X_train, y_train, X_test, y_test)

    assert list(searcher.val_indices_) == [16, 17, 18, 19]
    assert searcher.fit_indices_.max() < searcher.val_indices_.min()


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