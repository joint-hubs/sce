"""Shared utilities for diagnostics scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from scripts.run import (
    CONFIGS_DIR,
    _filter_target_rows,
    _resolve_context_variant,
    _resolve_categorical_mode,
    _resolve_standard_model,
    _run_sce_enrichment,
    _split_dataset,
    load_config,
    load_dataset,
    prepare_features,
    train_configured_model,
)


def evaluate_config_dataframe(
    config: dict[str, Any],
    config_name: str,
    df,
    target_col: str,
    use_cross_fitting_override: bool | None = None,
) -> dict[str, float]:
    """Run baseline + SCE on provided dataframe and return RMSE/R2 metrics."""
    context_variant = _resolve_context_variant(config)
    categorical_mode = _resolve_categorical_mode(config)
    model_type, model_params = _resolve_standard_model(config)

    if use_cross_fitting_override is not None:
        config = dict(config)
        config["sce"] = dict(config.get("sce", {}))
        config["sce"]["use_cross_fitting"] = use_cross_fitting_override

    split = _split_dataset(df, target_col, config=config)
    enrichment = _run_sce_enrichment(
        split.train_df,
        config,
        target_col,
        transform_df=split.test_df,
        context_variant=context_variant,
        categorical_mode=categorical_mode,
    )

    train_base = prepare_features(split.train_df, config, target_col)
    test_base = prepare_features(
        split.test_df,
        config,
        target_col,
        encoder=train_base.encoder,
        droplist=train_base.droplist,
    )

    baseline_model = train_configured_model(train_base.X.fillna(0), train_base.y, model_type, model_params)
    baseline_preds = baseline_model.predict(test_base.X.fillna(0))
    baseline_rmse = float(np.sqrt(mean_squared_error(test_base.y, baseline_preds)))
    baseline_r2 = float(r2_score(test_base.y, baseline_preds))

    train_sce = prepare_features(
        enrichment.df_enriched.loc[split.train_idx],
        config,
        target_col,
        encoder=train_base.encoder,
        droplist=train_base.droplist,
    )
    test_sce = prepare_features(
        enrichment.df_enriched.loc[split.test_idx],
        config,
        target_col,
        encoder=train_base.encoder,
        droplist=train_base.droplist,
    )

    train_sce_X = train_sce.X.copy()
    test_sce_X = test_sce.X.copy()
    for col in enrichment.sce_feature_cols:
        if col in enrichment.df_enriched.columns:
            train_sce_X[col] = enrichment.df_enriched.loc[train_sce_X.index, col]
            test_sce_X[col] = enrichment.df_enriched.loc[test_sce_X.index, col]

    sce_model = train_configured_model(train_sce_X.fillna(0), train_sce.y, model_type, model_params)
    sce_preds = sce_model.predict(test_sce_X.fillna(0))
    sce_rmse = float(np.sqrt(mean_squared_error(test_sce.y, sce_preds)))
    sce_r2 = float(r2_score(test_sce.y, sce_preds))

    return {
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2,
        "sce_rmse": sce_rmse,
        "sce_r2": sce_r2,
    }


def load_config_and_dataset(config_name: str):
    config_path = CONFIGS_DIR / f"{config_name}.toml"
    config = load_config(config_path)
    df = load_dataset(config)
    target_col = config["target"]["column"]
    # Same target hygiene as run_experiment: rows without a target are unusable
    df = _filter_target_rows(df, target_col)
    return config, config_path, df, target_col
