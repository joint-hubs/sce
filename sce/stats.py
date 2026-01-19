"""
@module: sce.stats
@depends: sce.config
@exports: StatsAggregator, compute_aggregations
@paper_ref: Equations 3.1-3.4
@data_flow: grouped_df -> aggregated_statistics -> enriched_features
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from sce.config import AggregationMethod

logger = logging.getLogger(__name__)


class StatsAggregator:
    """
    Computes statistical aggregations for hierarchical groups.
    
    Implements the core statistical operators from Equations 3.1-3.4 in the paper.
    Each aggregation produces summary statistics for a group (Eq. 3.1: φ^(k)(x_t)).
    """
    
    # Standard aggregation functions mapping
    # NOTE: STD uses ddof=0 (population standard deviation) intentionally.
    # Rationale: In SCE, we compute statistics for the ENTIRE group/neighborhood,
    # not a sample from a larger population. The group IS the population for that
    # hierarchy level. Using ddof=1 would underestimate variance for small groups.
    # For consistency with numpy.std() default and population-level interpretation.
    _STANDARD_AGGS = {
        # Central tendency
        AggregationMethod.MEAN: "mean",
        AggregationMethod.MEDIAN: "median",
        
        # Dispersion measures
        AggregationMethod.STD: lambda x: x.std(ddof=0),  # Population std (see note above)
        AggregationMethod.VAR: lambda x: x.var(ddof=0),  # Population variance
        AggregationMethod.CV: lambda x: x.std(ddof=0) / (x.mean() + 1e-8),  # Coefficient of variation
        AggregationMethod.IQR: lambda x: x.quantile(0.75) - x.quantile(0.25),  # Interquartile range
        
        # Quantiles/Percentiles (Paper: "quantiles")
        AggregationMethod.Q05: lambda x: x.quantile(0.05),
        AggregationMethod.Q10: lambda x: x.quantile(0.10),
        AggregationMethod.Q20: lambda x: x.quantile(0.20),
        AggregationMethod.Q33: lambda x: x.quantile(0.33),
        AggregationMethod.Q25: lambda x: x.quantile(0.25),
        AggregationMethod.Q66: lambda x: x.quantile(0.66),
        AggregationMethod.Q75: lambda x: x.quantile(0.75),
        AggregationMethod.Q80: lambda x: x.quantile(0.80),
        AggregationMethod.Q90: lambda x: x.quantile(0.90),
        AggregationMethod.Q95: lambda x: x.quantile(0.95),
        
        # Range
        AggregationMethod.MIN: "min",
        AggregationMethod.MAX: "max",
        AggregationMethod.RANGE: lambda x: x.max() - x.min(),
        
        # Counts
        AggregationMethod.COUNT: "count",
        AggregationMethod.SUM: "sum",
    }
    
    def __init__(self, methods: List[AggregationMethod]):
        """
        Initialize aggregator with specified methods.
        
        Args:
            methods: List of aggregation methods to compute
        """
        self.methods = methods
    
    def aggregate(
        self,
        df: pd.DataFrame,
        group_cols: List[str],
        value_col: str,
        min_size: int = 5
    ) -> pd.DataFrame:
        """
        Compute aggregations for each group.
        
        Implements the summarizer S_k from Equation 3.1:
            φ^(k)(x_t) = S_k({y_s : s ∈ N_k(t)})
        
        Args:
            df: Input dataframe
            group_cols: Columns to group by (defines neighborhoods N_k)
            value_col: Column to aggregate (target variable y)
            min_size: Minimum group size for valid statistics
            
        Returns:
            DataFrame with aggregated statistics per group
        """
        if not group_cols:
            # Global statistics (no grouping)
            logger.debug(f"Computing global statistics for '{value_col}'")
            return self._compute_global_stats(df, value_col)
        
        # Check if all group columns exist
        missing = [col for col in group_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing group columns: {missing}")
            raise ValueError(f"Missing group columns: {missing}")
        
        if value_col not in df.columns:
            logger.error(f"Value column '{value_col}' not found")
            raise ValueError(f"Value column '{value_col}' not found")
        
        logger.debug(f"Computing aggregations for groups: {group_cols}")
        
        # Build aggregation dict
        agg_dict = {}
        for method in self.methods:
            func = self._STANDARD_AGGS[method]
            agg_dict[f"{value_col}_{method.value}"] = (value_col, func)
        
        # Compute group statistics
        grouped = df.groupby(group_cols, dropna=False)
        stats = grouped.agg(**agg_dict).reset_index()
        
        initial_groups = len(stats)
        logger.debug(f"  Initial groups: {initial_groups}")
        
        # Filter small groups (use count if available)
        count_col = f"{value_col}_count"
        if AggregationMethod.COUNT in self.methods and count_col in stats.columns:
            stats = stats[stats[count_col] >= min_size].copy()
            filtered_groups = len(stats)
            if filtered_groups < initial_groups:
                logger.debug(f"  Filtered to {filtered_groups} groups (min_size={min_size}), "
                           f"removed {initial_groups - filtered_groups} small groups")
        elif len(stats) > 0:
            # Manually compute counts for filtering if COUNT not requested
            group_counts = df.groupby(group_cols, dropna=False).size()
            valid_groups = group_counts[group_counts >= min_size].index
            if len(group_cols) == 1:
                stats = stats[stats[group_cols[0]].isin(valid_groups)].copy()
            else:
                merge_df = pd.DataFrame(valid_groups.tolist(), columns=group_cols)
                stats = stats.merge(merge_df, on=group_cols, how='inner')
            
            filtered_groups = len(stats)
            if filtered_groups < initial_groups:
                logger.debug(f"  Filtered to {filtered_groups} groups (min_size={min_size}), "
                           f"removed {initial_groups - filtered_groups} small groups")
        
        return stats
    
    def _compute_global_stats(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Compute global (dataset-level) statistics."""
        result = {}
        values = df[value_col].dropna()
        
        for method in self.methods:
            func = self._STANDARD_AGGS.get(method)
            if func is None:
                continue  # Skip unknown methods
            if callable(func):
                result[f"{value_col}_{method.value}"] = func(values)
            else:
                result[f"{value_col}_{method.value}"] = getattr(values, str(func))()
        
        return pd.DataFrame([result])


def compute_aggregations(
    df: pd.DataFrame,
    categorical_cols: List[str],
    target_col: str,
    methods: List[AggregationMethod],
    min_group_size: int = 5,
    include_global: bool = True,
    include_interactions: bool = True,
    max_interaction_depth: int = 2,
    # DEPRECATED parameters for backward compatibility
    hierarchy: Optional[List[str]] = None,
    additional_categorical_cols: Optional[List[str]] = None,
    include_quantiles: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Compute aggregations across all categorical columns.
    
    For each categorical column (and optionally 2-way combinations), computes
    group statistics (mean, median, quantiles, etc.) of the target variable.
    
    Args:
        df: Input dataframe
        categorical_cols: Categorical columns for grouping (auto-detected or specified)
        target_col: Target variable name
        methods: Aggregation methods to apply (use AggregationMethod enum)
        min_group_size: Minimum samples per group
        include_global: Whether to include global (dataset-wide) statistics
        include_interactions: Whether to compute 2-way interactions between categoricals
        max_interaction_depth: Maximum number of columns to combine (2 = pairs only)
        hierarchy: DEPRECATED - use categorical_cols instead
        additional_categorical_cols: DEPRECATED - use categorical_cols instead
        include_quantiles: DEPRECATED - use AggregationMethod.Q25/Q75 in methods list
        
    Returns:
        Dict mapping group_key → aggregated statistics DataFrame
        
    Example:
        >>> from sce.config import AggregationMethod
        >>> stats = compute_aggregations(
        ...     df, 
        ...     categorical_cols=["city", "room_type", "is_superhost"],
        ...     target_col="price",
        ...     methods=[AggregationMethod.MEAN, AggregationMethod.Q25, AggregationMethod.Q75],
        ...     include_interactions=True
        ... )
        >>> stats.keys()  # ['global', 'city', 'room_type', 'is_superhost', 'city__room_type', ...]
    """
    import warnings
    from itertools import combinations
    
    aggregator = StatsAggregator(methods)
    stats_dict: Dict[str, pd.DataFrame] = {}
    
    # Handle deprecated parameters
    all_categoricals = list(categorical_cols) if categorical_cols else []
    
    if hierarchy is not None:
        warnings.warn(
            "hierarchy parameter is deprecated. Use categorical_cols instead.",
            DeprecationWarning,
            stacklevel=2
        )
        all_categoricals = list(dict.fromkeys(all_categoricals + list(hierarchy)))
    
    if additional_categorical_cols is not None:
        warnings.warn(
            "additional_categorical_cols is deprecated. Use categorical_cols instead.",
            DeprecationWarning,
            stacklevel=2
        )
        all_categoricals = list(dict.fromkeys(all_categoricals + list(additional_categorical_cols)))
    
    # Filter to columns that exist
    all_categoricals = [c for c in all_categoricals if c in df.columns]
    
    logger.info(f"Computing aggregations for {len(all_categoricals)} categorical columns")
    logger.debug(f"Categorical columns: {all_categoricals}")
    logger.debug(f"Aggregation methods: {[m.value for m in methods]}")
    logger.debug(f"min_group_size={min_group_size}, include_global={include_global}, "
                f"include_interactions={include_interactions}")
    
    # Global statistics (level 0)
    if include_global:
        logger.debug("Computing global statistics")
        stats_dict["global"] = aggregator.aggregate(df, [], target_col, min_group_size)
    
    # Single-column aggregations
    logger.debug(f"Computing single-column aggregations for {len(all_categoricals)} columns")
    for cat_col in all_categoricals:
        level_name = cat_col
        if level_name not in stats_dict:
            try:
                stats_df = aggregator.aggregate(df, [cat_col], target_col, min_group_size)
                stats_dict[level_name] = stats_df
                logger.debug(f"  '{cat_col}': {len(stats_df)} groups, {len(stats_df.columns)} features")
            except ValueError as e:
                logger.warning(f"  Failed to compute stats for '{cat_col}': {e}")
                continue
    
    # 2-way (and higher) interactions
    if include_interactions and len(all_categoricals) >= 2:
        interaction_count = 0
        logger.debug(f"Computing interactions (max depth: {max_interaction_depth})")
        for depth in range(2, min(max_interaction_depth + 1, len(all_categoricals) + 1)):
            for combo in combinations(all_categoricals, depth):
                group_cols = list(combo)
                level_name = "__".join(group_cols)
                
                if level_name in stats_dict:
                    continue
                
                try:
                    stats_df = aggregator.aggregate(df, group_cols, target_col, min_group_size)
                    stats_dict[level_name] = stats_df
                    interaction_count += 1
                    logger.debug(f"  '{level_name}': {len(stats_df)} groups")
                except ValueError as e:
                    logger.debug(f"  Failed interaction '{level_name}': {e}")
                    continue
        
        logger.info(f"Created {interaction_count} interaction levels")
    
    logger.info(f"Total hierarchy levels created: {len(stats_dict)}")
    
    # DEPRECATED: include_quantiles - emit warning if used
    if include_quantiles:
        warnings.warn(
            "include_quantiles is deprecated. Use AggregationMethod.Q25/Q75 in methods list instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Backward compatibility: add Q25/Q75 if not already computed
        if AggregationMethod.Q25 not in methods or AggregationMethod.Q75 not in methods:
            stats_dict = _add_quantile_stats(df, stats_dict, categorical_cols, target_col)
    
    return stats_dict


def compute_relative_features(
    df: pd.DataFrame,
    stats_dict: Dict[str, pd.DataFrame],
    target_col: str,
    categorical_cols: Optional[List[str]] = None,
    # DEPRECATED
    hierarchy: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute relative features from group statistics (OPTIMIZED).
    
    Implements Equation 3 from the paper:
        r_{k,z}(x_t) = (y_t - μ_k) / (σ_k + ε)      # Z-score
        r_{k,ratio}(x_t) = y_t / (median_k + ε)     # Ratio to median
        r_{k,pct}(x_t) = (y_t - q25) / (q75 - q25)  # Percentile position
    
    WARNING: These features use y_t (target value) in their formula, 
    causing direct target leakage. Only use for post-hoc analysis.
    
    Performance: Uses vectorized numpy operations for speed.
    
    Args:
        df: Input dataframe with joined statistics
        stats_dict: Precomputed group statistics
        target_col: Target variable name
        categorical_cols: Categorical column names
        hierarchy: DEPRECATED - use categorical_cols
        
    Returns:
        DataFrame with relative features added
    """
    import warnings
    
    # Handle deprecated parameter
    if hierarchy is not None and categorical_cols is None:
        warnings.warn("hierarchy parameter is deprecated. Use categorical_cols.", DeprecationWarning)
        categorical_cols = hierarchy
    
    # Pre-extract target values as numpy array for vectorized ops
    target_vals = df[target_col].values
    epsilon = 1e-8
    
    # Collect all new columns in dict (faster than iterative assignment)
    new_cols: Dict[str, np.ndarray] = {}
    
    # Process all levels in stats_dict
    for level_name in stats_dict.keys():
        if level_name == "global":
            continue  # Skip global for relative features
        
        # Column name patterns
        mean_col = f"{level_name}_{target_col}_mean"
        std_col = f"{level_name}_{target_col}_std"
        median_col = f"{level_name}_{target_col}_median"
        q25_col = f"{level_name}_{target_col}_q25"
        q75_col = f"{level_name}_{target_col}_q75"
        
        # Z-score relative feature (Eq. 3): (y - μ) / (σ + ε)
        if mean_col in df.columns and std_col in df.columns:
            mean_vals = df[mean_col].values
            std_vals = df[std_col].values
            new_cols[f"{level_name}_zscore"] = (
                (target_vals - mean_vals) / (std_vals + epsilon)
            )
        
        # Ratio to median (Eq. 3): y / (median + ε)
        if median_col in df.columns:
            median_vals = df[median_col].values
            new_cols[f"{level_name}_ratio"] = (
                target_vals / (median_vals + epsilon)
            )
        
        # Percentile position: (y - Q25) / (IQR + ε) — normalized position in distribution
        if q25_col in df.columns and q75_col in df.columns:
            q25_vals = df[q25_col].values
            q75_vals = df[q75_col].values
            iqr_vals = q75_vals - q25_vals
            new_cols[f"{level_name}_pct_position"] = (
                (target_vals - q25_vals) / (iqr_vals + epsilon)
            )
        
        # Deviation from mean (absolute, for interpretability)
        if mean_col in df.columns:
            mean_vals = df[mean_col].values
            new_cols[f"{level_name}_dev_from_mean"] = target_vals - mean_vals
    
    # Single DataFrame construction (much faster than iterative assignment)
    if new_cols:
        new_df = pd.DataFrame(new_cols, index=df.index)
        result = pd.concat([df, new_df], axis=1)
    else:
        result = df.copy()
    
    return result


def _get_level_names(hierarchy: List[str]) -> List[str]:
    """Generate level names from hierarchy."""
    return ["__".join(hierarchy[:k+1]) for k in range(len(hierarchy))]


def _level_name_to_cols(level_name: str) -> List[str]:
    """
    Convert level name back to column list.
    
    Uses '__' as delimiter to avoid conflicts with column names containing '_'.
    """
    return level_name.split("__")


def _compute_cardinalities(df: pd.DataFrame, categorical_cols: List[str]) -> Dict[str, int]:
    """Compute cardinalities for categorical columns."""
    return {col: df[col].nunique(dropna=False) for col in categorical_cols if col in df.columns}


def build_fallback_chains(
    stats_dict: Dict[str, pd.DataFrame],
    categorical_cols: List[str],
    cardinalities: Dict[str, int],
) -> Dict[str, List[str]]:
    """
    Build ordered fallback chains for each hierarchy level.
    """
    from itertools import combinations

    fallback_chains: Dict[str, List[str]] = {}
    levels = [level for level in stats_dict.keys() if level != "global"]

    for level_name in levels:
        level_cols = _level_name_to_cols(level_name)
        if len(level_cols) <= 1:
            chain = ["global"] if "global" in stats_dict else []
            fallback_chains[level_name] = chain
            continue

        fallbacks: List[str] = []
        for r in range(len(level_cols) - 1, 0, -1):
            subsets = list(combinations(level_cols, r))
            subsets.sort(key=lambda s: sum(cardinalities.get(c, 0) for c in s))
            for subset in subsets:
                subset_name = "__".join(subset)
                if subset_name in stats_dict and subset_name not in fallbacks:
                    fallbacks.append(subset_name)

        if "global" in stats_dict:
            fallbacks.append("global")

        fallback_chains[level_name] = fallbacks

    return fallback_chains


def apply_hierarchical_backoff(
    df: pd.DataFrame,
    stats_dict: Dict[str, pd.DataFrame],
    target_col: str,
    categorical_cols: Optional[List[str]] = None,
    # DEPRECATED
    hierarchy: Optional[List[str]] = None,
    add_backoff_depth: bool = False,
) -> pd.DataFrame:
    """
    Apply backoff for missing statistics.
    
    Per the paper (Section 3.4): "if a fine-grained group has too few samples
    to produce a stable estimate, we can back off to a higher-level grouping
    or shrink the group statistic towards a global value."
    
    This function fills NaN values with values from global stats as fallback.
    
    Args:
        df: DataFrame with joined statistics (may have NaN)
        stats_dict: Dictionary of statistics at each level
        target_col: Target column name
        categorical_cols: List of categorical column names
        hierarchy: DEPRECATED - use categorical_cols
        
    Returns:
        DataFrame with NaN values filled via backoff to global
    """
    import warnings
    
    result = df.copy()
    
    # Handle deprecated parameter
    if hierarchy is not None and categorical_cols is None:
        warnings.warn("hierarchy parameter is deprecated. Use categorical_cols.", DeprecationWarning)
        categorical_cols = hierarchy

    # Get list of aggregation suffixes from global stats
    agg_suffixes = []
    if "global" in stats_dict:
        global_stats = stats_dict["global"]
        for col in global_stats.columns:
            if col.startswith(target_col):
                suffix = col[len(target_col):]  # e.g., "_mean"
                agg_suffixes.append(suffix)

    if not agg_suffixes:
        return result

    cardinalities = _compute_cardinalities(result, categorical_cols or [])
    chains = build_fallback_chains(stats_dict, categorical_cols or [], cardinalities)

    original_means: Dict[str, pd.Series] = {}
    if add_backoff_depth:
        for level_name in chains.keys():
            mean_col = f"{level_name}_{target_col}_mean"
            if mean_col in result.columns:
                original_means[level_name] = result[mean_col].copy()
    
    # For each aggregation type, fill NaN using hierarchical fallback chains
    for suffix in agg_suffixes:
        global_col = f"global_{target_col}{suffix}"
        for level_name, fallbacks in chains.items():
            col_name = f"{level_name}_{target_col}{suffix}"
            if col_name not in result.columns:
                continue

            mask = result[col_name].isna()
            for fallback_level in fallbacks:
                if not mask.any():
                    break

                if fallback_level == "global":
                    fallback_col = global_col
                else:
                    fallback_col = f"{fallback_level}_{target_col}{suffix}"

                if fallback_col in result.columns:
                    result.loc[mask, col_name] = result.loc[mask, fallback_col]
                    mask = result[col_name].isna()

    if add_backoff_depth and original_means:
        for level_name, fallbacks in chains.items():
            mean_col = f"{level_name}_{target_col}_mean"
            if mean_col not in result.columns:
                continue
            original = original_means.get(level_name)
            if original is None:
                continue

            depth_col = f"{level_name}_backoff_depth"
            depth = pd.Series(0, index=result.index, dtype="int64")
            remaining = original.isna().copy()

            for depth_idx, fallback_level in enumerate(fallbacks, start=1):
                if not remaining.any():
                    break

                if fallback_level == "global":
                    fallback_col = f"global_{target_col}_mean"
                else:
                    fallback_col = f"{fallback_level}_{target_col}_mean"

                if fallback_col in result.columns:
                    fill_mask = remaining & result[fallback_col].notna()
                    depth.loc[fill_mask] = depth_idx
                    remaining = remaining & ~result[fallback_col].notna()

            depth.loc[remaining] = -1
            result[depth_col] = depth
    
    return result


def _add_quantile_stats(
    df: pd.DataFrame,
    stats_dict: Dict[str, pd.DataFrame],
    categorical_cols: List[str],
    target_col: str
) -> Dict[str, pd.DataFrame]:
    """
    Add Q25 and Q75 percentiles to statistics.
    
    DEPRECATED: Use AggregationMethod.Q25 and AggregationMethod.Q75 in methods list instead.
    This function is kept for backward compatibility only.
    """
    if not categorical_cols:
        return stats_dict
    
    # Only add quantiles to single-column aggregations for simplicity
    for col in categorical_cols:
        if col not in stats_dict:
            continue
        
        stats_df = stats_dict[col]
        
        # Compute quantiles
        grouped = df.groupby([col], dropna=False)[target_col]
        q25 = grouped.quantile(0.25).rename(f"{target_col}_q25")
        q75 = grouped.quantile(0.75).rename(f"{target_col}_q75")
        
        # Join to existing stats
        stats_df = stats_df.merge(
            pd.DataFrame({f"{target_col}_q25": q25, f"{target_col}_q75": q75}),
            left_on=[col],
            right_index=True,
            how="left"
        )
        stats_dict[col] = stats_df
    
    return stats_dict
