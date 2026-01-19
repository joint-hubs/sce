"""
@module: sce.pipeline
@depends: sce.engine, sce.config
@exports: fit_context_pipeline, create_sce_pipeline
@paper_ref: Section 6 (Implementation)
@data_flow: raw_df -> sce_features -> model -> predictions
"""

from typing import Any, Optional

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from sce.config import ContextConfig
from sce.engine import StatisticalContextEngine


def create_sce_pipeline(
    config: ContextConfig,
    model: Optional[BaseEstimator] = None
) -> Pipeline:
    """
    Create a scikit-learn pipeline with SCE feature engineering.
    
    Args:
        config: SCE configuration
        model: Optional model to add at the end (e.g., XGBoost, LightGBM)
        
    Returns:
        sklearn Pipeline with SCE transformer and optional model
        
    Example:
        >>> from xgboost import XGBRegressor
        >>> from sce import ContextConfig, create_sce_pipeline
        >>> 
        >>> config = ContextConfig(hierarchy=["region", "city"], target_col="price")
        >>> pipeline = create_sce_pipeline(config, model=XGBRegressor())
        >>> pipeline.fit(train_df, train_df["price"])
    """
    steps = [("sce", StatisticalContextEngine(config))]
    
    if model is not None:
        steps.append(("model", model))
    
    return Pipeline(steps)


def fit_context_pipeline(
    df: pd.DataFrame,
    config: ContextConfig,
    model: Optional[BaseEstimator] = None,
    **fit_params: Any
) -> Pipeline:
    """
    Convenience function to fit a complete SCE pipeline.
    
    Args:
        df: Training dataframe
        config: SCE configuration
        model: Optional model to include
        **fit_params: Additional parameters for model fitting
        
    Returns:
        Fitted pipeline
    """
    pipeline = create_sce_pipeline(config, model)
    
    if model is not None:
        y = df[config.target_col]
        X = df.drop(columns=[config.target_col])
        pipeline.fit(X, y, **fit_params)
    else:
        pipeline.fit(df)
    
    return pipeline
