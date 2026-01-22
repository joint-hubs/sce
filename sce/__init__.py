"""
@module: sce
@depends:
@exports: StatisticalContextEngine, ContextConfig, fit_context_pipeline, detect_categorical_columns
@paper_ref: Algorithm 1
@data_flow: public API imports
"""

from sce.cleanup import FeatureCleanupPipeline
from sce.config import AggregationMethod, CleanupConfig, ContextConfig, detect_categorical_columns
from sce.engine import StatisticalContextEngine
from sce.importance import aggregate_importance, run_iterative_pruning
from sce.model_presets import load_xgboost_presets, resolve_xgboost_presets
from sce.pipeline import fit_context_pipeline
from sce.search import FeatureCombinationSearch, SearchResult, SearchSummary
from sce.selection import LMFeatureSelector, compute_lm_statistics, select_significant_features

__version__ = "0.3.3"  # Citation and author info update
__all__ = [
    # Core
    "StatisticalContextEngine",
    "ContextConfig",
    "AggregationMethod",
    "CleanupConfig",
    "detect_categorical_columns",
    "fit_context_pipeline",
    "FeatureCleanupPipeline",
    # Feature selection
    "LMFeatureSelector",
    "compute_lm_statistics",
    "select_significant_features",
    # Model search
    "FeatureCombinationSearch",
    "SearchResult",
    "SearchSummary",
    # Model presets
    "load_xgboost_presets",
    "resolve_xgboost_presets",
    # Importance + pruning
    "aggregate_importance",
    "run_iterative_pruning",
]
