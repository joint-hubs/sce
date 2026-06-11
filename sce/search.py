"""
@module: sce.search
@depends: numpy, pandas, sklearn, sce.model_presets, sce.models
@exports: FeatureCombinationSearch, SearchResult, SearchSummary, train_model
@paper_ref: Section 4.3 Model Selection
@data_flow: feature subsets -> model presets -> trained estimators -> ranked search results
@status: EXPERIMENTAL - Test coverage 21%. Not recommended for production use.

Random search over feature combinations with multiple model configurations.
Implements the comprehensive search from sce_analysis.py with cross-fitting.

⚠️ WARNING: This module has minimal test coverage (21%) and is considered
experimental. Use at your own risk. Core SCE functionality in engine.py
and stats.py is fully tested and production-ready.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sce.model_presets import load_model_presets
from sce.models import build_model, extract_feature_importance


@dataclass
class SearchResult:
    """Result from a single model configuration.

    ``eval_set`` records which holdout produced the headline metrics:
    candidate configurations are scored on the internal validation split
    (``eval_set="validation"``), while the selected winners are refit on the
    full training data and scored exactly once on the test set
    (``eval_set="test"``). For test-evaluated results, ``val_rmse``/``val_r2``/
    ``val_mae`` preserve the validation score that drove the selection.
    """

    config_id: int
    strategy: str  # 'baseline', 'context_only', 'base_context'
    n_features: int
    n_base: int
    n_context: int
    rmse: float
    r2: float
    mae: float
    features: List[str]
    model_config: str  # e.g., 'default', 'shallow', 'boosted'
    feature_importance: Optional[pd.DataFrame] = None
    eval_set: str = "validation"
    val_rmse: Optional[float] = None
    val_r2: Optional[float] = None
    val_mae: Optional[float] = None


@dataclass
class SearchSummary:
    """Summary of model search results.

    ``all_results`` holds every candidate scored on the validation split.
    ``best_by_rmse``, ``best_by_r2`` and ``baseline_result`` are selected on
    validation metrics, refit on the full training data, and carry unbiased
    test-set metrics (``eval_set="test"``).
    """

    all_results: List[SearchResult]
    best_by_rmse: SearchResult
    best_by_r2: SearchResult
    baseline_result: SearchResult
    feature_usage: Dict[str, int]  # feature -> count of appearances in top models


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_type: str = "xgboost",
    config_name: str = "default",
    model_params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[Any, Dict[str, float], pd.DataFrame]:
    """
    Train a model and compute metrics.

    Args:
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        model_type: Supported downstream model type.
        config_name: Model configuration name

    Returns:
        Tuple of (model, metrics_dict, feature_importance_df)
    """
    # Clean data (no imputation)
    X_train = X_train.copy().replace([np.inf, -np.inf], np.nan)
    X_test = X_test.copy().replace([np.inf, -np.inf], np.nan)

    train_mask = ~X_train.isna().any(axis=1)
    test_mask = ~X_test.isna().any(axis=1)
    X_train = X_train.loc[train_mask]
    y_train = y_train.loc[train_mask]
    X_test = X_test.loc[test_mask]
    y_test = y_test.loc[test_mask]
    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("No rows left after dropping missing values")

    config_source = model_params or load_model_presets(model_type)
    params = dict(config_source.get(config_name, config_source["default"]))

    model = build_model(model_type, params)
    model.fit(X_train, y_train)

    importance = extract_feature_importance(model, X_train.columns)

    # Predictions and metrics
    y_pred = model.predict(X_test)

    metrics = {
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
    }

    return model, metrics, importance


class FeatureCombinationSearch:
    """
    Search over random feature combinations.

    Implements the sampling strategy from sce_analysis.py:
    - Sample 5% of all 2^n combinations (min=50, max=500)
    - Test with multiple model configurations
    - Track feature importance across all models

    Strategies tested:
    - baseline: base features only
    - context_only: SCE features only (random subsets)
    - context_only_all: ALL SCE features
    - base_context: base + random SCE subsets
    - base_context_all: base + ALL SCE features
    - base_context_sig_lm: base + LM p-value significant SCE features
    - base_context_sig_tree: base + tree-importance significant SCE features
    - ablation_remove_best: all features, iteratively remove most important
    - ablation_remove_worst: all features, iteratively remove least important
    """

    def __init__(
        self,
        base_features: List[str],
        context_features: List[str],
        sampling_pct: float = 5.0,
        min_samples: int = 50,
        max_samples: int = 500,
        model_configs: List[str] = None,
        model_params: Optional[Dict[str, Dict[str, Any]]] = None,
        model_type: str = "xgboost",
        random_state: int = 42,
        run_ablation: bool = True,
        run_significance_selection: bool = True,
        p_threshold: float = 0.1,
        val_fraction: float = 0.2,
        val_strategy: str = "random",
    ):
        """
        Initialize search.

        Args:
            base_features: Traditional features (always included in base+context)
            context_features: SCE features to sample combinations from
            sampling_pct: Percentage of 2^n combinations to sample
            min_samples: Minimum number of configurations to test
            max_samples: Maximum number of configurations to test
            model_configs: List of model config names to test
            model_type: Supported downstream model type.
            random_state: Random seed for reproducibility
            run_ablation: Whether to run ablation experiments (remove best/worst)
            run_significance_selection: Whether to run LM/tree significance selection
            p_threshold: P-value threshold for LM significance selection
            val_fraction: Fraction of the training rows held out as the internal
                validation split used for candidate selection
            val_strategy: "random" for a shuffled validation split, "tail" to
                hold out the last rows in the given order (use with
                time-ordered training data to avoid temporal leakage)
        """
        self.base_features = base_features
        self.context_features = context_features
        self.sampling_pct = sampling_pct
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.model_configs = model_configs or list(load_model_presets(model_type).keys())
        self.model_params = model_params
        self.model_type = model_type
        self.random_state = random_state
        self.run_ablation = run_ablation
        self.run_significance_selection = run_significance_selection
        self.p_threshold = p_threshold
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        if val_strategy not in {"random", "tail"}:
            raise ValueError("val_strategy must be 'random' or 'tail'")
        self.val_fraction = val_fraction
        self.val_strategy = val_strategy

        self.results_: List[SearchResult] = []
        self.feature_gains_: Dict[str, List[float]] = {}
        self.fit_indices_: Optional[np.ndarray] = None
        self.val_indices_: Optional[np.ndarray] = None

    def _generate_combinations(self) -> List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]]:
        """Generate random feature combinations to test.

        Strategies tested (matching old pipeline):
        1. baseline: base features only (no SCE)
        2. context_only: random SCE feature subsets (no base)
        3. base_context: base + random SCE subsets
        4. base_context_all: base + ALL SCE features
        """
        np.random.seed(self.random_state)

        n_context = len(self.context_features)
        n_possible = 2**n_context if n_context > 0 else 1
        n_samples = max(self.min_samples, int(n_possible * self.sampling_pct / 100))
        n_samples = min(n_samples, self.max_samples, n_possible)

        combinations: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = []
        sampled: Set[Tuple[str, ...]] = set()

        # =====================================================================
        # 1. Baseline: base features only (no context)
        # =====================================================================
        combinations.append(("baseline", tuple(self.base_features), ()))

        # =====================================================================
        # 2. Context-only combinations (no base features)
        # =====================================================================
        if n_context > 0:
            # 2a. All context features
            combinations.append(("context_only_all", (), tuple(self.context_features)))

            # 2b. Random context subsets
            n_context_only = max(10, n_samples // 5)
            for _ in range(n_context_only):
                n = np.random.randint(1, n_context + 1)
                subset = tuple(sorted(np.random.choice(self.context_features, n, replace=False)))
                if subset not in sampled:
                    sampled.add(subset)
                    combinations.append(("context_only", (), subset))

        # =====================================================================
        # 3. Base + ALL context features
        # =====================================================================
        if n_context > 0:
            combinations.append(
                ("base_context_all", tuple(self.base_features), tuple(self.context_features))
            )

        # =====================================================================
        # 4. Base + random context subsets
        # =====================================================================
        for _ in range(n_samples):
            if n_context == 0:
                continue
            n = np.random.randint(1, n_context + 1)
            subset = tuple(sorted(np.random.choice(self.context_features, n, replace=False)))
            if subset not in sampled:
                sampled.add(subset)
                combinations.append(("base_context", tuple(self.base_features), subset))

        return combinations

    def _split_selection_indices(self, n_rows: int) -> Tuple[np.ndarray, np.ndarray]:
        """Split training row positions into fit/validation for candidate selection."""
        n_val = max(1, int(round(n_rows * self.val_fraction)))
        if n_val >= n_rows:
            raise ValueError(
                f"val_fraction={self.val_fraction} leaves no fit rows for n_rows={n_rows}"
            )

        if self.val_strategy == "tail":
            fit_idx = np.arange(n_rows - n_val)
            val_idx = np.arange(n_rows - n_val, n_rows)
        else:
            rng = np.random.RandomState(self.random_state)
            perm = rng.permutation(n_rows)
            val_idx = np.sort(perm[:n_val])
            fit_idx = np.sort(perm[n_val:])
        return fit_idx, val_idx

    def search(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SearchSummary:
        """
        Run the feature combination search.

        Evaluation protocol: the training data is split into an internal
        fit/validation partition (see ``val_fraction``/``val_strategy``).
        Every candidate combination is scored on the validation split only.
        The winners (best by validation RMSE, best by validation R2, and the
        best baseline) are then refit on the full training data and evaluated
        exactly once on the test set, so reported test metrics are free of
        selection bias.

        Args:
                X_train: Training features
                y_train: Training targets
                X_test: Test features (used only for the final evaluation)
                y_test: Test targets (used only for the final evaluation)
            progress_callback: Optional callback(current, total) for progress

        Returns:
            SearchSummary with validation-scored candidates and test-scored winners
        """
        combinations = self._generate_combinations()

        fit_idx, val_idx = self._split_selection_indices(len(X_train))
        self.fit_indices_ = fit_idx
        self.val_indices_ = val_idx
        X_fit = X_train.iloc[fit_idx]
        y_fit = y_train.iloc[fit_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        # =================================================================
        # Add significance-based selection strategies (fit split only —
        # the validation split must stay untouched during candidate design)
        # =================================================================
        if self.run_significance_selection and len(self.context_features) > 0:
            sig_combos = self._generate_significance_combinations(X_fit, y_fit)
            combinations.extend(sig_combos)

        # =================================================================
        # Add ablation strategies (remove best/worst iteratively)
        # =================================================================
        if self.run_ablation and len(self.context_features) > 0:
            ablation_combos = self._generate_ablation_combinations(X_fit, y_fit)
            combinations.extend(ablation_combos)

        total = len(combinations) * len(self.model_configs)

        self.results_ = []
        self.feature_gains_ = {}
        config_id = 0
        common_feature_names = set(X_train.columns).intersection(X_test.columns)

        for strategy, base_subset, context_subset in combinations:
            features = list(base_subset) + list(context_subset)
            features = [f for f in features if f in common_feature_names]

            if not features:
                continue

            X_fit_sub = X_fit[features]
            X_val_sub = X_val[features]

            for model_cfg in self.model_configs:
                try:
                    model, metrics, importance = train_model(
                        X_fit_sub,
                        y_fit,
                        X_val_sub,
                        y_val,
                        self.model_type,
                        model_cfg,
                        model_params=self.model_params,
                    )

                    result = SearchResult(
                        config_id=config_id,
                        strategy=strategy,
                        n_features=len(features),
                        n_base=len([f for f in features if f in self.base_features]),
                        n_context=len([f for f in features if f in self.context_features]),
                        rmse=metrics["rmse"],
                        r2=metrics["r2"],
                        mae=metrics["mae"],
                        features=features,
                        model_config=model_cfg,
                        feature_importance=importance,
                        eval_set="validation",
                        val_rmse=metrics["rmse"],
                        val_r2=metrics["r2"],
                        val_mae=metrics["mae"],
                    )
                    self.results_.append(result)

                    # Track feature importance
                    if model_cfg == "default":
                        for _, row in importance.iterrows():
                            feat = row["feature"]
                            if feat not in self.feature_gains_:
                                self.feature_gains_[feat] = []
                            self.feature_gains_[feat].append(row["importance"])

                except Exception as e:
                    # Log the failure so we can debug
                    import logging

                    logging.getLogger(__name__).debug(
                        "Config %s strategy=%s model=%s failed: %s",
                        config_id,
                        strategy,
                        model_cfg,
                        e,
                    )

                config_id += 1
                if progress_callback:
                    progress_callback(config_id, total)

        # Guard: if no results, raise with helpful message
        if not self.results_:
            raise RuntimeError(
                f"Search produced no valid results. "
                f"Tested {total} combinations but all failed. "
                f"Check debug logs for train_model failures."
            )

        # Selection happens on validation metrics only
        valid_r2 = [r for r in self.results_ if np.isfinite(r.r2)]
        best_val_rmse = min(self.results_, key=lambda r: r.rmse)
        best_val_r2 = max(valid_r2 or self.results_, key=lambda r: r.r2)
        baseline_candidates = [r for r in self.results_ if r.strategy == "baseline"]
        best_val_baseline = (
            min(baseline_candidates, key=lambda r: r.rmse) if baseline_candidates else None
        )

        # Final evaluation: refit winners on the full training data, score
        # once on the test set. Cache so identical winners are not retrained.
        test_eval_cache: Dict[Tuple[Tuple[str, ...], str], SearchResult] = {}

        def _evaluate_on_test(selected: SearchResult) -> SearchResult:
            cache_key = (tuple(selected.features), selected.model_config)
            if cache_key in test_eval_cache:
                return test_eval_cache[cache_key]

            _, metrics, importance = train_model(
                X_train[selected.features],
                y_train,
                X_test[selected.features],
                y_test,
                self.model_type,
                selected.model_config,
                model_params=self.model_params,
            )
            final = SearchResult(
                config_id=selected.config_id,
                strategy=selected.strategy,
                n_features=selected.n_features,
                n_base=selected.n_base,
                n_context=selected.n_context,
                rmse=metrics["rmse"],
                r2=metrics["r2"],
                mae=metrics["mae"],
                features=selected.features,
                model_config=selected.model_config,
                feature_importance=importance,
                eval_set="test",
                val_rmse=selected.rmse,
                val_r2=selected.r2,
                val_mae=selected.mae,
            )
            test_eval_cache[cache_key] = final
            return final

        best_rmse = _evaluate_on_test(best_val_rmse)
        best_r2 = _evaluate_on_test(best_val_r2)
        baseline_result = (
            _evaluate_on_test(best_val_baseline) if best_val_baseline is not None else best_rmse
        )

        # Feature usage in top 20 models (by validation RMSE)
        top_20 = sorted(self.results_, key=lambda r: r.rmse)[:20]
        feature_usage = {}
        for r in top_20:
            for f in r.features:
                feature_usage[f] = feature_usage.get(f, 0) + 1

        return SearchSummary(
            all_results=self.results_,
            best_by_rmse=best_rmse,
            best_by_r2=best_r2,
            baseline_result=baseline_result,
            feature_usage=feature_usage,
        )

    def _generate_significance_combinations(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]]:
        """Generate combinations using significance-based feature selection.

        Strategies:
        - base_context_sig_lm: base + LM p-value significant SCE features
        - base_context_sig_tree: base + tree-importance top SCE features
        """
        from scipy import stats
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        combinations: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = []
        context_in_X = [f for f in self.context_features if f in X_train.columns]

        if not context_in_X:
            return combinations

        # =================================================================
        # LM p-value significance selection
        # =================================================================
        try:
            X_ctx = X_train[context_in_X].copy().fillna(0)
            X_ctx = X_ctx.replace([np.inf, -np.inf], 0)

            # Fit LM to get p-values
            model = LinearRegression()
            model.fit(X_ctx, y_train)

            n = len(y_train)
            k = len(context_in_X)
            y_pred = model.predict(X_ctx)
            residuals = y_train - y_pred
            mse = np.sum(residuals**2) / max(n - k - 1, 1)

            X_with_const = np.column_stack([np.ones(n), X_ctx.values])
            try:
                var_coef = mse * np.linalg.pinv(X_with_const.T @ X_with_const).diagonal()
                se = np.sqrt(np.maximum(var_coef[1:], 1e-10))
                t_stats = model.coef_ / se
                p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), max(n - k - 1, 1)))
            except Exception:
                p_values = np.ones(k)

            # Select significant features
            sig_mask = p_values < self.p_threshold
            sig_features = [context_in_X[i] for i in range(len(context_in_X)) if sig_mask[i]]

            if sig_features:
                combinations.append(
                    ("base_context_sig_lm", tuple(self.base_features), tuple(sig_features))
                )
                # Also context-only with significant features
                combinations.append(("context_only_sig_lm", (), tuple(sig_features)))
        except Exception:
            pass

        # =================================================================
        # Tree importance significance selection
        # =================================================================
        try:
            X_ctx = X_train[context_in_X].copy().fillna(0)
            X_ctx = X_ctx.replace([np.inf, -np.inf], 0)

            rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
            rf.fit(X_ctx, y_train)

            importance = pd.DataFrame(
                {"feature": context_in_X, "importance": rf.feature_importances_}
            ).sort_values("importance", ascending=False)

            # Top 50% by importance
            n_top = max(1, len(context_in_X) // 2)
            top_features = importance.head(n_top)["feature"].tolist()

            if top_features:
                combinations.append(
                    ("base_context_sig_tree", tuple(self.base_features), tuple(top_features))
                )
                combinations.append(("context_only_sig_tree", (), tuple(top_features)))
        except Exception:
            pass

        return combinations

    def _generate_ablation_combinations(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]]:
        """Generate ablation combinations (remove best/worst features iteratively).

        Strategies:
        - ablation_remove_best_N: remove top N most important features
        - ablation_remove_worst_N: remove bottom N least important features
        """
        from sklearn.ensemble import RandomForestRegressor

        combinations: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = []
        all_features = list(self.base_features) + list(self.context_features)
        all_in_X = [f for f in all_features if f in X_train.columns]

        if len(all_in_X) < 3:
            return combinations

        try:
            X_all = X_train[all_in_X].copy().fillna(0)
            X_all = X_all.replace([np.inf, -np.inf], 0)

            rf = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
            rf.fit(X_all, y_train)

            importance = pd.DataFrame(
                {"feature": all_in_X, "importance": rf.feature_importances_}
            ).sort_values("importance", ascending=False)

            ordered_features = importance["feature"].tolist()

            # Ablation: remove most important (1, 2, 3, 5, 10 features)
            for n_remove in [1, 2, 3, 5, 10]:
                if n_remove >= len(ordered_features):
                    continue
                remaining = ordered_features[n_remove:]
                base_remaining = tuple(f for f in remaining if f in self.base_features)
                ctx_remaining = tuple(f for f in remaining if f in self.context_features)
                combinations.append(
                    (f"ablation_remove_best_{n_remove}", base_remaining, ctx_remaining)
                )

            # Ablation: remove least important (1, 2, 3, 5, 10 features)
            for n_remove in [1, 2, 3, 5, 10]:
                if n_remove >= len(ordered_features):
                    continue
                remaining = ordered_features[:-n_remove]
                base_remaining = tuple(f for f in remaining if f in self.base_features)
                ctx_remaining = tuple(f for f in remaining if f in self.context_features)
                combinations.append(
                    (f"ablation_remove_worst_{n_remove}", base_remaining, ctx_remaining)
                )

            # Progressive ablation: keep only top N features
            for n_keep in [3, 5, 10, 20]:
                if n_keep >= len(ordered_features):
                    continue
                top_n = ordered_features[:n_keep]
                base_top = tuple(f for f in top_n if f in self.base_features)
                ctx_top = tuple(f for f in top_n if f in self.context_features)
                combinations.append((f"ablation_top_{n_keep}_only", base_top, ctx_top))

        except Exception:
            pass

        return combinations

    def get_aggregated_importance(self) -> pd.DataFrame:
        """Get feature importance aggregated across all models."""
        if not self.feature_gains_:
            return pd.DataFrame()

        agg = []
        for feat, gains in self.feature_gains_.items():
            agg.append(
                {
                    "feature": feat,
                    "mean_importance": np.mean(gains),
                    "std_importance": np.std(gains),
                    "n_appearances": len(gains),
                    "is_context": feat in self.context_features,
                }
            )

        return pd.DataFrame(agg).sort_values("mean_importance", ascending=False)
