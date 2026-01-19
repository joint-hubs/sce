"""
@module: sce.config
@depends:
@exports: ContextConfig, AggregationMethod, detect_categorical_columns
@paper_ref: Section 3.1
@data_flow: user config -> validated parameters
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, List, Literal, Optional

if TYPE_CHECKING:
    import pandas as pd


class AggregationMethod(Enum):
    """Statistical aggregation methods for context features.

    Paper Section 3.1: "means, medians, dispersion measures, quantiles,
    counts, and relative deviations"
    """

    # Central tendency
    MEAN = "mean"
    MEDIAN = "median"

    # Dispersion measures
    STD = "std"
    VAR = "var"
    CV = "cv"  # Coefficient of variation (std/mean)
    IQR = "iqr"  # Interquartile range (Q75-Q25)

    # Quantiles/Percentiles
    Q05 = "q05"  # 5th percentile
    Q10 = "q10"  # 10th percentile
    Q20 = "q20"  # 20th percentile
    Q33 = "q33"  # 33rd percentile
    Q25 = "q25"  # 25th percentile (first quartile)
    Q66 = "q66"  # 66th percentile
    Q75 = "q75"  # 75th percentile (third quartile)
    Q80 = "q80"  # 80th percentile
    Q90 = "q90"  # 90th percentile
    Q95 = "q95"  # 95th percentile

    # Range
    MIN = "min"
    MAX = "max"
    RANGE = "range"  # max - min

    # Counts
    COUNT = "count"
    SUM = "sum"


@dataclass
class CleanupConfig:
    """
    Configuration for feature cleanup pipeline.
    """

    leakage_enabled: bool = True
    leakage_remove_threshold: float = 0.95
    leakage_warn_threshold: float = 0.85

    correlation_enabled: bool = True
    correlation_threshold: float = 0.9
    correlation_method: Literal["pearson", "spearman"] = "pearson"
    correlation_drop_strategy: Literal[
        "lower_target_corr",
        "lower_variance",
        "first",
        "hierarchy",
    ] = "lower_target_corr"
    correlation_max_iterations: int = 1000

    vif_enabled: bool = False
    vif_threshold: float = 10.0
    vif_max_iterations: int = 100

    hierarchy_enabled: bool = True
    hierarchy_corr_threshold: float = 0.95
    hierarchy_prefer: Literal["child", "parent"] = "child"

    min_variance: float = 1e-10


def detect_categorical_columns(
    df: "pd.DataFrame",
    target_col: str,
    max_cardinality: int = 100,
    min_cardinality: int = 2,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Auto-detect categorical columns suitable for SCE grouping.

    Detection rules:
    1. Object or category dtype → categorical
    2. Boolean dtype → categorical
    3. Integer with low cardinality (≤ max_cardinality) → likely categorical
    4. Exclude target column and any specified exclusions
    5. Exclude columns with only 1 unique value (no variance)

    Args:
        df: Input DataFrame
        target_col: Target column name (will be excluded)
        max_cardinality: Maximum unique values for a column to be considered categorical
        min_cardinality: Minimum unique values (must have at least 2 groups)
        exclude_cols: Additional columns to exclude

    Returns:
        List of detected categorical column names

    Example:
        >>> categoricals = detect_categorical_columns(df, target_col="price")
        >>> print(categoricals)
        ['city', 'room_type', 'property_type', 'is_superhost']
    """
    import pandas as pd

    exclude = set(exclude_cols or [])
    exclude.add(target_col)

    categoricals = []

    for col in df.columns:
        if col in exclude:
            continue

        n_unique = df[col].nunique()

        # Skip if no variance (only 1 unique value)
        if n_unique < min_cardinality:
            continue

        # Skip if too many unique values (likely continuous or ID)
        if n_unique > max_cardinality:
            continue

        dtype = df[col].dtype

        # Object or category dtype → categorical
        if dtype == "object" or dtype.name == "category":
            categoricals.append(col)
        # Boolean → categorical
        elif dtype == "bool":
            categoricals.append(col)
        # Low cardinality integer (likely encoded categorical)
        elif pd.api.types.is_integer_dtype(dtype):
            # Additional check: cardinality should be small relative to data size
            if n_unique <= max_cardinality and n_unique < len(df) * 0.1:
                categoricals.append(col)

    return categoricals


@dataclass
class ContextConfig:
    """
    Configuration for Statistical Context Engineering.

    Supports two modes:
    1. **Auto-detection mode** (recommended): Set `categorical_cols=None` and the engine
       will auto-detect categorical columns from the DataFrame.
    2. **Manual mode**: Specify `categorical_cols` explicitly.

    Attributes:
        target_col: Name of the target variable column (REQUIRED)
        categorical_cols: List of categorical columns for grouping. If None, auto-detected.
        min_categorical_columns: Minimum number of categorical columns required to run SCE
        aggregations: List of aggregation methods to apply
        min_group_size: Minimum samples required per group
        use_cross_fitting: Whether to apply out-of-fold aggregation (prevents leakage)
        n_folds: Number of folds for cross-fitting
        include_fold_variance: Whether to add fold-variance uncertainty features
        fold_variance_features: Which variance features to include (std/lower/upper/cv)
        include_relative_features: Whether to compute z-score/ratio features.
            WARNING: These features use y_t (the target value) in their formula,
            causing direct target leakage. Only enable for post-hoc analysis.
        include_global_stats: Whether to include dataset-wide global statistics
        include_interactions: Whether to compute 2-way categorical interactions
        max_interaction_depth: Maximum number of columns to combine (2 = pairs only)
        max_cardinality: For auto-detection, max unique values to consider categorical
        exclude_cols: Columns to exclude from auto-detection
        add_backoff_depth: Whether to add backoff depth features
        cleanup_config: Optional feature cleanup configuration

    Example (auto-detection):
        config = ContextConfig(
            target_col="price",
            include_interactions=True
        )
        # Categorical columns detected automatically from DataFrame

    Example (manual):
        config = ContextConfig(
            target_col="price",
            categorical_cols=["city", "room_type", "is_superhost"],
            include_interactions=True
        )
    """

    target_col: str
    categorical_cols: Optional[List[str]] = None  # None = auto-detect
    min_categorical_columns: int = 1
    aggregations: List[AggregationMethod] = field(
        default_factory=lambda: [
            AggregationMethod.MEAN,
            AggregationMethod.MEDIAN,
            AggregationMethod.STD,
            AggregationMethod.Q05,
            AggregationMethod.Q20,
            AggregationMethod.Q80,
            AggregationMethod.Q95,
            AggregationMethod.COUNT,
        ]
    )
    min_group_size: int = 5
    use_cross_fitting: bool = True
    n_folds: int = 5
    include_fold_variance: bool = True
    fold_variance_features: List[str] = field(default_factory=lambda: ["std", "lower", "upper"])
    include_relative_features: bool = False  # WARNING: Causes target leakage!
    include_global_stats: bool = True  # Global (dataset-wide) statistics
    include_interactions: bool = True  # 2-way categorical interactions (default ON now)
    max_interaction_depth: int = 2  # Only pairs (A×B), not triples (A×B×C)
    max_cardinality: int = 100  # For auto-detection
    exclude_cols: List[str] = field(default_factory=list)  # Exclude from auto-detection
    add_backoff_depth: bool = False
    cleanup_config: Optional[CleanupConfig] = None

    # DEPRECATED: kept for backward compatibility
    hierarchy: Optional[List[str]] = None
    additional_categorical_cols: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if not self.target_col:
            raise ValueError("target_col must be specified")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2")
        if self.min_group_size < 1:
            raise ValueError("min_group_size must be at least 1")
        if self.min_categorical_columns < 0:
            raise ValueError("min_categorical_columns must be at least 0")

        allowed_variance_features = {"std", "lower", "upper", "cv"}
        if any(v not in allowed_variance_features for v in self.fold_variance_features):
            raise ValueError(
                "fold_variance_features must be a subset of {'std','lower','upper','cv'}"
            )

        # Backward compatibility: merge hierarchy + additional_categorical_cols into categorical_cols
        if self.hierarchy is not None:
            import warnings

            warnings.warn(
                "ContextConfig.hierarchy is deprecated. Use categorical_cols instead. "
                "All categorical columns are now treated equally (no ordered hierarchy).",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.categorical_cols is None:
                self.categorical_cols = list(self.hierarchy)
            else:
                # Merge without duplicates
                self.categorical_cols = list(
                    dict.fromkeys(list(self.categorical_cols) + list(self.hierarchy))
                )

        if self.additional_categorical_cols is not None:
            import warnings

            warnings.warn(
                "ContextConfig.additional_categorical_cols is deprecated. "
                "Use categorical_cols instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if self.categorical_cols is None:
                self.categorical_cols = list(self.additional_categorical_cols)
            else:
                self.categorical_cols = list(
                    dict.fromkeys(
                        list(self.categorical_cols) + list(self.additional_categorical_cols)
                    )
                )

    def get_categorical_cols(self, df: "pd.DataFrame") -> List[str]:
        """
        Get categorical columns for grouping.

        If categorical_cols is None, auto-detects from DataFrame.
        Otherwise returns the specified columns (filtered to those that exist).

        Args:
            df: Input DataFrame

        Returns:
            List of categorical column names
        """
        if self.categorical_cols is None:
            # Auto-detect
            return detect_categorical_columns(
                df=df,
                target_col=self.target_col,
                max_cardinality=self.max_cardinality,
                exclude_cols=self.exclude_cols,
            )
        else:
            # Manual: filter to columns that exist in df
            return [c for c in self.categorical_cols if c in df.columns]
