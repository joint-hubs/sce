"""
@module: sce.engine
@depends: sce.config, sce.stats, sce.meta
@exports: StatisticalContextEngine
@paper_ref: Algorithm 1, Equations 3.1-3.4
@data_flow: raw_df -> enriched_df -> model_ready_features
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold

from sce.cleanup import FeatureCleanupPipeline
from sce.config import ContextConfig
from sce.meta import component
from sce.stats import (
    _level_name_to_cols,
    apply_hierarchical_backoff,
    compute_aggregations,
    compute_relative_features,
)

logger = logging.getLogger(__name__)


@component(
    name="StatisticalContextEngine",
    responsibility="Core engine for categorical feature enrichment with auto-detection",
    depends_on=["ContextConfig", "StatsAggregator"]
)
class StatisticalContextEngine(BaseEstimator, TransformerMixin):
    """
    Statistical Context Engineering transformer.
    
    Implements Algorithm 1 from the paper: enriches datasets with statistical
    context features using out-of-fold aggregation to prevent leakage.
    
    Supports auto-detection of categorical columns or manual specification.
    
    Compatible with scikit-learn pipelines via fit/transform interface.
    
    Example (auto-detection):
        >>> from sce import StatisticalContextEngine, ContextConfig
        >>> config = ContextConfig(target_col="price", use_cross_fitting=True)
        >>> engine = StatisticalContextEngine(config)
        >>> enriched_df = engine.fit_transform(train_df)  # Auto-detects categoricals
        
    Example (manual):
        >>> config = ContextConfig(
        ...     target_col="price",
        ...     categorical_cols=["city", "room_type", "is_superhost"]
        ... )
        >>> engine = StatisticalContextEngine(config)
        >>> enriched_df = engine.fit_transform(train_df)
    """
    
    def __init__(self, config: ContextConfig):
        """
        Initialize the context engine.
        
        Args:
            config: Configuration specifying target and aggregation settings
        """
        self.config = config
        self._stats_dict: Optional[Dict[str, pd.DataFrame]] = None
        self._categorical_cols: Optional[List[str]] = None  # Detected/resolved columns
        self._fitted = False
        self._cleanup_pipeline: Optional[FeatureCleanupPipeline] = None
        self._cleanup_removed_features: Optional[List[str]] = None
        self._cleanup_report = None
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "StatisticalContextEngine":
        """
        Fit the context engine (learn group statistics).
        
        Implements Algorithm 1 Step 1: Compute group summaries.
        Auto-detects categorical columns if not specified in config.
        
        Args:
            X: Input dataframe containing categorical columns and target
            y: Ignored (for sklearn compatibility)
            
        Returns:
            self
            
        Raises:
            ValueError: If required columns are missing
        """
        logger.debug(f"Fitting SCE engine on {len(X)} rows, {len(X.columns)} columns")
        self._validate_input(X)
        
        # Get categorical columns (auto-detect or from config)
        self._categorical_cols = self.config.get_categorical_cols(X)

        min_required = self.config.min_categorical_columns
        if min_required > 0 and len(self._categorical_cols) < min_required:
            logger.error(
                "Insufficient categorical columns for SCE: required=%s found=%s cols=%s",
                min_required,
                len(self._categorical_cols),
                self._categorical_cols,
            )
            raise ValueError(
                f"SCE requires at least {min_required} categorical columns after filtering. "
                f"Found {len(self._categorical_cols)}."
            )
        
        if self._categorical_cols:
            logger.info(f"SCE using {len(self._categorical_cols)} categorical columns: {self._categorical_cols}")
            print(f"  SCE using {len(self._categorical_cols)} categorical columns: {self._categorical_cols}")
            
            # Log cardinality of each categorical column
            for col in self._categorical_cols:
                if col in X.columns:
                    unique_count = X[col].nunique()
                    logger.debug(f"  Column '{col}': {unique_count} unique values")
        else:
            logger.warning("No categorical columns detected or specified")
        
        # Compute statistics for all categorical columns
        logger.debug(f"Computing aggregations with methods: {[m.value for m in self.config.aggregations]}")
        logger.debug(f"min_group_size={self.config.min_group_size}, "
                    f"include_interactions={self.config.include_interactions}, "
                    f"max_interaction_depth={self.config.max_interaction_depth}")
        
        self._stats_dict = compute_aggregations(
            df=X,
            categorical_cols=self._categorical_cols,
            target_col=self.config.target_col,
            methods=self.config.aggregations,
            min_group_size=self.config.min_group_size,
            include_global=self.config.include_global_stats,
            include_interactions=self.config.include_interactions,
            max_interaction_depth=self.config.max_interaction_depth
        )
        
        logger.info(f"Computed statistics for {len(self._stats_dict)} hierarchy levels")
        for level_name, stats_df in self._stats_dict.items():
            logger.debug(f"  Level '{level_name}': {len(stats_df)} groups, {len(stats_df.columns)} features")
        
        self._fitted = True
        logger.debug("SCE engine fit complete")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform dataframe by adding context features.
        
        Implements Algorithm 1 Step 2: Join summaries to dataset.
        
        Args:
            X: Input dataframe with categorical columns
            
        Returns:
            Enriched dataframe with context features added
            
        Raises:
            RuntimeError: If fit() has not been called
        """
        if not self._fitted or self._stats_dict is None:
            raise RuntimeError("Must call fit() before transform()")
        
        logger.debug(f"Transforming {len(X)} rows")
        self._validate_input(X)
        
        # Start with copy of input (preserves index)
        result = X.copy()
        original_index = X.index
        initial_col_count = len(result.columns)
        
        # Join statistics for each categorical level
        for level_name, stats_df in self._stats_dict.items():
            logger.debug(f"Joining statistics for level '{level_name}'")
            if level_name == "global":
                # Broadcast global stats to all rows
                result = self._join_global_stats(result, stats_df)
                logger.debug(f"  Added {len(result.columns) - initial_col_count} global features")
            else:
                # Join stats by group keys
                group_cols = _level_name_to_cols(level_name)
                before_cols = len(result.columns)
                result = self._join_level_stats(result, stats_df, group_cols, level_name)
                added_cols = len(result.columns) - before_cols
                logger.debug(f"  Level '{level_name}' (groups: {group_cols}): added {added_cols} features")
        
        features_before_backoff = len(result.columns)
        
        # Apply backoff for small groups (Paper Section 3.4)
        logger.debug("Applying hierarchical backoff for small groups")
        result = apply_hierarchical_backoff(
            df=result,
            stats_dict=self._stats_dict,
            categorical_cols=self._categorical_cols,
            target_col=self.config.target_col,
            add_backoff_depth=self.config.add_backoff_depth,
        )
        
        # Add relative features ONLY if explicitly enabled
        # WARNING: Relative features cause target leakage (use y_t in formula)
        if self.config.include_relative_features:
            logger.debug("Computing relative features (may cause leakage)")
            result = compute_relative_features(
                df=result,
                stats_dict=self._stats_dict,
                categorical_cols=self._categorical_cols,
                target_col=self.config.target_col
            )
        else:
            logger.debug("Relative features disabled (no target leakage)")

        # Optional feature cleanup
        if self.config.cleanup_config is not None:
            if self._cleanup_pipeline is None:
                self._cleanup_pipeline = FeatureCleanupPipeline(self.config.cleanup_config)

            if self._cleanup_removed_features is None:
                if self.config.target_col in result.columns:
                    features = result.drop(columns=[self.config.target_col])
                    _, report = self._cleanup_pipeline.fit_transform(
                        features,
                        result[self.config.target_col],
                        target_col=self.config.target_col,
                    )
                    self._cleanup_removed_features = self._cleanup_pipeline.removed_features_
                    self._cleanup_report = report
                else:
                    logger.warning(
                        "Cleanup enabled but target column missing; skipping cleanup fit"
                    )
            if self._cleanup_removed_features:
                result = result.drop(columns=self._cleanup_removed_features, errors="ignore")
        
        # Ensure original index is preserved
        result.index = original_index
        
        total_added = len(result.columns) - initial_col_count
        logger.info(f"Transform complete: added {total_added} enriched features")
        
        return result
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit and transform in one step (with cross-fitting if enabled).
        
        Implements Algorithm 1 with Equation 4 (out-of-fold aggregation).
        
        Args:
            X: Input dataframe
            y: Ignored
            
        Returns:
            Enriched dataframe with leakage-safe context features
        """
        if self.config.use_cross_fitting:
            # Use cross-fitting for leakage-safe context
            return self._fit_transform_cross_fitted(X)
        else:
            # Standard fit-transform (may leak in-sample)
            return self.fit(X, y).transform(X)
    
    def _fit_transform_cross_fitted(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform using out-of-fold aggregation.
        
        Implements Equation 4 from the paper:
            φ^(k)_{cf}(x_t) = S_k({y_s : s ∈ N_k(t) ∩ (indices \\ fold_m)})
        
        Each fold gets statistics computed from all OTHER folds.
        
        Args:
            X: Input dataframe
            
        Returns:
            Enriched dataframe with out-of-fold context features
        """
        self._validate_input(X)
        
        # Get categorical columns (auto-detect or from config)
        self._categorical_cols = self.config.get_categorical_cols(X)
        
        # Store original index
        original_index = X.index
        
        # Reset index to avoid duplicate index issues during concat
        X_reset = X.reset_index(drop=True)
        
        kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)
        fold_indices = list(kf.split(X_reset))
        fold_results = {}

        # Phase 1: compute stats for each fold separately
        per_fold_stats: List[Dict[str, pd.DataFrame]] = []
        for _, val_idx in fold_indices:
            fold_df = X_reset.iloc[val_idx]
            fold_stats = compute_aggregations(
                df=fold_df,
                categorical_cols=self._categorical_cols,
                target_col=self.config.target_col,
                methods=self.config.aggregations,
                min_group_size=self.config.min_group_size,
                include_global=self.config.include_global_stats,
                include_interactions=self.config.include_interactions,
                max_interaction_depth=self.config.max_interaction_depth,
            )
            per_fold_stats.append(fold_stats)

        # Phase 2: aggregate stats from other folds for each validation fold
        for current_fold, (_, val_idx) in enumerate(fold_indices):
            other_fold_stats = [
                stats for i, stats in enumerate(per_fold_stats) if i != current_fold
            ]
            stats_dict = self._aggregate_fold_statistics(
                other_fold_stats,
                include_variance=self.config.include_fold_variance,
                variance_features=self.config.fold_variance_features,
            )

            val_df = X_reset.iloc[val_idx].copy()

            for level_name, stats_df in stats_dict.items():
                if level_name == "global":
                    val_df = self._join_global_stats(val_df, stats_df)
                else:
                    group_cols = _level_name_to_cols(level_name)
                    val_df = self._join_level_stats(val_df, stats_df, group_cols, level_name)

            val_df = apply_hierarchical_backoff(
                df=val_df,
                stats_dict=stats_dict,
                categorical_cols=self._categorical_cols,
                target_col=self.config.target_col,
                add_backoff_depth=self.config.add_backoff_depth,
            )

            if self.config.include_relative_features:
                val_df = compute_relative_features(
                    df=val_df,
                    stats_dict=stats_dict,
                    categorical_cols=self._categorical_cols,
                    target_col=self.config.target_col,
                )

            for idx, row_idx in enumerate(val_idx):
                fold_results[row_idx] = val_df.iloc[idx]
        
        # Reconstruct in original order
        enriched = pd.DataFrame([fold_results[i] for i in range(len(X_reset))])
        
        # Restore original index
        enriched.index = original_index
        
        # Store final stats for future transform calls
        if self.config.include_fold_variance:
            self._stats_dict = self._aggregate_fold_statistics(
                per_fold_stats,
                include_variance=True,
                variance_features=self.config.fold_variance_features,
            )
        else:
            self._stats_dict = compute_aggregations(
                df=X,
                categorical_cols=self._categorical_cols,
                target_col=self.config.target_col,
                methods=self.config.aggregations,
                min_group_size=self.config.min_group_size,
                include_global=self.config.include_global_stats,
                include_interactions=self.config.include_interactions,
                max_interaction_depth=self.config.max_interaction_depth,
            )
        self._fitted = True

        if self.config.cleanup_config is not None:
            if self._cleanup_pipeline is None:
                self._cleanup_pipeline = FeatureCleanupPipeline(self.config.cleanup_config)
            if self._cleanup_removed_features is None and self.config.target_col in enriched.columns:
                features = enriched.drop(columns=[self.config.target_col])
                _, report = self._cleanup_pipeline.fit_transform(
                    features,
                    enriched[self.config.target_col],
                    target_col=self.config.target_col,
                )
                self._cleanup_removed_features = self._cleanup_pipeline.removed_features_
                self._cleanup_report = report
            if self._cleanup_removed_features:
                enriched = enriched.drop(columns=self._cleanup_removed_features, errors="ignore")

        return enriched
    
    def _join_global_stats(self, df: pd.DataFrame, stats_df: pd.DataFrame) -> pd.DataFrame:
        """Broadcast global statistics to all rows."""
        result = df.copy()
        for col in stats_df.columns:
            result[f"global_{col}"] = stats_df.iloc[0][col]
        return result

    def _aggregate_fold_statistics(
        self,
        fold_stats_list: List[Dict[str, pd.DataFrame]],
        include_variance: bool,
        variance_features: List[str],
    ) -> Dict[str, pd.DataFrame]:
        """Aggregate statistics from multiple folds into point estimates + variance."""
        if not fold_stats_list:
            return {}

        result: Dict[str, pd.DataFrame] = {}
        for level_name in fold_stats_list[0].keys():
            level_dfs = [fs.get(level_name) for fs in fold_stats_list if fs.get(level_name) is not None]
            if not level_dfs:
                continue

            if level_name == "global":
                result[level_name] = self._aggregate_global_level(
                    level_dfs,
                    include_variance,
                    variance_features,
                )
            else:
                group_cols = _level_name_to_cols(level_name)
                result[level_name] = self._aggregate_group_level(
                    level_dfs,
                    group_cols,
                    include_variance,
                    variance_features,
                )

        return result

    def _aggregate_group_level(
        self,
        level_dfs: List[pd.DataFrame],
        group_cols: List[str],
        include_variance: bool,
        variance_features: List[str],
    ) -> pd.DataFrame:
        """Aggregate hierarchical group statistics across folds."""
        if not level_dfs:
            return pd.DataFrame()

        stat_cols = [c for c in level_dfs[0].columns if c not in group_cols]
        all_groups = pd.concat([df[group_cols] for df in level_dfs], ignore_index=True).drop_duplicates()
        result = all_groups.copy()

        for stat_col in stat_cols:
            is_count = stat_col.endswith("_count")
            values_df = all_groups.copy()

            for i, df in enumerate(level_dfs):
                if stat_col not in df.columns:
                    continue
                fold_df = df[group_cols + [stat_col]].copy()
                fold_df = fold_df.rename(columns={stat_col: f"_fold_{i}"})
                values_df = values_df.merge(fold_df, on=group_cols, how="left")

            fold_cols = [c for c in values_df.columns if c.startswith("_fold_")]
            if not fold_cols:
                continue

            if is_count:
                # Count reflects total samples used across folds (sum of fold counts)
                result[stat_col] = values_df[fold_cols].sum(axis=1, skipna=True)
            else:
                result[stat_col] = values_df[fold_cols].mean(axis=1, skipna=True)

            if include_variance and not is_count and len(fold_cols) >= 2:
                fold_std = values_df[fold_cols].std(axis=1, ddof=1).fillna(0)
                fold_mean = result[stat_col]

                if "std" in variance_features:
                    result[f"{stat_col}_fold_std"] = fold_std
                if "cv" in variance_features:
                    result[f"{stat_col}_fold_cv"] = fold_std / (fold_mean.abs() + 1e-8)
                if "lower" in variance_features:
                    result[f"{stat_col}_fold_lower"] = fold_mean - 2 * fold_std
                if "upper" in variance_features:
                    result[f"{stat_col}_fold_upper"] = fold_mean + 2 * fold_std

        return result

    def _aggregate_global_level(
        self,
        level_dfs: List[pd.DataFrame],
        include_variance: bool,
        variance_features: List[str],
    ) -> pd.DataFrame:
        """Aggregate global statistics across folds."""
        if not level_dfs:
            return pd.DataFrame()

        stat_cols = level_dfs[0].columns.tolist()
        result_data: Dict[str, float] = {}

        for stat_col in stat_cols:
            is_count = stat_col.endswith("_count")
            fold_values = [df[stat_col].iloc[0] for df in level_dfs if stat_col in df.columns]
            if not fold_values:
                continue

            if is_count:
                # Sum fold counts to represent total samples used for stats
                point = float(np.nansum(fold_values))
            else:
                point = float(np.nanmean(fold_values))
            result_data[stat_col] = point

            if include_variance and not is_count and len(fold_values) >= 2:
                fold_std = float(np.nanstd(fold_values, ddof=1))
                if "std" in variance_features:
                    result_data[f"{stat_col}_fold_std"] = fold_std
                if "cv" in variance_features:
                    result_data[f"{stat_col}_fold_cv"] = fold_std / (abs(point) + 1e-8)
                if "lower" in variance_features:
                    result_data[f"{stat_col}_fold_lower"] = point - 2 * fold_std
                if "upper" in variance_features:
                    result_data[f"{stat_col}_fold_upper"] = point + 2 * fold_std

        return pd.DataFrame([result_data])
    
    def _join_level_stats(
        self,
        df: pd.DataFrame,
        stats_df: pd.DataFrame,
        group_cols: list[str],
        level_name: str
    ) -> pd.DataFrame:
        """Join hierarchical statistics by group keys."""
        # Ensure join columns have compatible types
        df_joined = df.copy()
        stats_joined = stats_df.copy()
        
        for col in group_cols:
            if col in df_joined.columns and col in stats_joined.columns:
                df_joined[col] = df_joined[col].astype(str)
                stats_joined[col] = stats_joined[col].astype(str)
        
        # Rename stat columns with prefix
        stat_cols = [c for c in stats_joined.columns if c not in group_cols]
        rename_map = {c: f"{level_name}_{c}" for c in stat_cols}
        stats_joined = stats_joined.rename(columns=rename_map)
        
        # Left join to preserve all input rows
        result = df_joined.merge(stats_joined, on=group_cols, how="left")
        
        return result
    
    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validate input dataframe has required columns."""
        if self.config.target_col not in X.columns:
            raise ValueError(f"Missing target column: {self.config.target_col}")
        
        # If categorical_cols specified manually, check they exist
        if self.config.categorical_cols is not None:
            missing_cols = [c for c in self.config.categorical_cols if c not in X.columns]
            if missing_cols:
                raise ValueError(f"Missing categorical columns: {missing_cols}")
