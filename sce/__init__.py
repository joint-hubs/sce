"""
@module: sce
@depends:
@exports: StatisticalContextEngine, ContextConfig, fit_context_pipeline, detect_categorical_columns
@paper_ref: Algorithm 1
@data_flow: public API imports
"""

from sce.baselines import (
    SUPPORTED_CONTEXT_VARIANTS,
    get_context_variant_label,
    resolve_context_variant_methods,
)
from sce.cleanup import FeatureCleanupPipeline
from sce.config import AggregationMethod, CleanupConfig, ContextConfig, detect_categorical_columns
from sce.engine import StatisticalContextEngine
from sce.importance import aggregate_importance, run_iterative_pruning
from sce.model_presets import (
    SUPPORTED_MODEL_TYPES,
    load_model_presets,
    load_xgboost_presets,
    resolve_model_presets,
    resolve_xgboost_presets,
)
from sce.models import build_model, get_model_label, model_supports_gpu
from sce.pipeline import create_sce_pipeline, fit_context_pipeline
from sce.search import FeatureCombinationSearch, SearchResult, SearchSummary
from sce.selection import LMFeatureSelector, compute_lm_statistics, select_significant_features

__version__ = "0.4.0"
__all__ = [
    # Core
    "StatisticalContextEngine",
    "ContextConfig",
    "AggregationMethod",
    "CleanupConfig",
    "detect_categorical_columns",
    "create_sce_pipeline",
    "fit_context_pipeline",
    "FeatureCleanupPipeline",
    "SUPPORTED_CONTEXT_VARIANTS",
    "get_context_variant_label",
    "resolve_context_variant_methods",
    # Feature selection
    "LMFeatureSelector",
    "compute_lm_statistics",
    "select_significant_features",
    # Model search
    "FeatureCombinationSearch",
    "SearchResult",
    "SearchSummary",
    # Model presets
    "SUPPORTED_MODEL_TYPES",
    "build_model",
    "get_model_label",
    "model_supports_gpu",
    "load_model_presets",
    "load_xgboost_presets",
    "resolve_model_presets",
    "resolve_xgboost_presets",
    # Importance + pruning
    "aggregate_importance",
    "run_iterative_pruning",
]
