"""
@module: sce.importance
@depends: pandas, numpy
@exports: aggregate_importance, run_iterative_pruning
@data_flow: search_results -> importance_stats -> pruning_trace
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sce.search import SearchResult, train_model


@dataclass
class PruningResult:
    step_pct_keep: float
    n_features: int
    rmse: float
    r2: float
    mae: float
    features: List[str]


def aggregate_importance(results: Iterable[SearchResult]) -> pd.DataFrame:
    """Aggregate feature importance statistics across model results."""
    rows: List[Dict[str, object]] = []

    for result in results:
        if result.feature_importance is None or result.feature_importance.empty:
            continue
        for _, row in result.feature_importance.iterrows():
            rows.append({
                "feature": row["feature"],
                "importance": float(row["importance"]),
                "strategy": result.strategy,
                "model_config": result.model_config,
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    agg_rows = []
    for feature, group in df.groupby("feature"):
        values = group["importance"].values
        agg_rows.append({
            "feature": feature,
            "n_models": len(values),
            "importance_mean": float(np.mean(values)),
            "importance_std": float(np.std(values)),
            "importance_min": float(np.min(values)),
            "importance_p10": float(np.percentile(values, 10)),
            "importance_p50": float(np.percentile(values, 50)),
            "importance_p90": float(np.percentile(values, 90)),
            "importance_max": float(np.max(values)),
            "strategies": "|".join(sorted(group["strategy"].unique())),
        })

    aggregated = pd.DataFrame(agg_rows).sort_values("importance_mean", ascending=False)

    # Add per-strategy mean importance
    strategy_means = df.groupby(["feature", "strategy"])["importance"].mean().unstack(fill_value=np.nan)
    aggregated = aggregated.merge(strategy_means, on="feature", how="left")

    return aggregated


def run_iterative_pruning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    features: List[str],
    model_config_name: str = "default",
    model_params: Optional[Dict[str, Dict[str, object]]] = None,
    step_pct_keep: Iterable[float] = (1.0, 0.8, 0.6, 0.4, 0.2),
) -> Tuple[List[PruningResult], pd.DataFrame]:
    """Run iterative pruning by keeping top-k% important features."""
    results: List[PruningResult] = []
    removed_records: List[Dict[str, object]] = []

    current_features = [f for f in features if f in X_train.columns]

    for pct in step_pct_keep:
        if not current_features:
            break
        keep_n = max(1, int(len(current_features) * pct))

        model, metrics, importance = train_model(
            X_train[current_features],
            y_train,
            X_test[current_features],
            y_test,
            model_type="xgboost",
            config_name=model_config_name,
            model_params=model_params,
        )

        ordered = importance.sort_values("importance", ascending=False)["feature"].tolist()
        kept = ordered[:keep_n]
        removed = [f for f in current_features if f not in kept]

        results.append(
            PruningResult(
                step_pct_keep=float(pct),
                n_features=len(kept),
                rmse=float(metrics["rmse"]),
                r2=float(metrics["r2"]),
                mae=float(metrics["mae"]),
                features=kept,
            )
        )

        for feat in removed:
            removed_records.append({
                "feature": feat,
                "removed_at_pct_keep": float(pct),
            })

        current_features = kept

    removed_df = pd.DataFrame(removed_records)
    return results, removed_df
