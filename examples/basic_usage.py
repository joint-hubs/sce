"""
Basic usage example for Statistical Context Engineering (SCE).

This example demonstrates:
1. Loading a dataset
2. Configuring SCE
3. Creating a pipeline with XGBoost
4. Training and evaluating
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

from sce import StatisticalContextEngine, ContextConfig, AggregationMethod
from sce.io import load_dataset


def main():
    # Load dataset (Poland short-term rentals)
    print("Loading dataset...")
    df = load_dataset("rental_poland_short")
    print(f"Dataset shape: {df.shape}")
    
    # Configure SCE
    config = ContextConfig(
        target_col="price",
        # categorical_cols auto-detected from DataFrame!
        aggregations=[
            AggregationMethod.MEAN,
            AggregationMethod.MEDIAN,
            AggregationMethod.STD,
            AggregationMethod.COUNT,
        ],
        use_cross_fitting=True,  # Prevents leakage
        n_folds=5,
        min_group_size=5,
    )
    
    # Create SCE engine
    engine = StatisticalContextEngine(config)
    
    # Split data
    X = df.drop(columns=["price"])
    y = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit SCE on training data and transform
    print("\nFitting SCE...")
    X_train_enriched = engine.fit_transform(X_train, y_train)
    X_test_enriched = engine.transform(X_test)
    
    print(f"Original features: {X_train.shape[1]}")
    print(f"Enriched features: {X_train_enriched.shape[1]}")
    print(f"Context features added: {X_train_enriched.shape[1] - X_train.shape[1]}")
    
    # Train baseline model (no SCE)
    print("\n--- Baseline (no SCE) ---")
    baseline_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    baseline_model.fit(X_train.select_dtypes(include=["number"]), y_train)
    baseline_preds = baseline_model.predict(X_test.select_dtypes(include=["number"]))
    baseline_rmse = mean_squared_error(y_test, baseline_preds, squared=False)
    baseline_r2 = r2_score(y_test, baseline_preds)
    print(f"RMSE: {baseline_rmse:.2f}")
    print(f"R²:   {baseline_r2:.4f}")
    
    # Train SCE-enriched model
    print("\n--- With SCE ---")
    sce_model = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    sce_model.fit(X_train_enriched.select_dtypes(include=["number"]), y_train)
    sce_preds = sce_model.predict(X_test_enriched.select_dtypes(include=["number"]))
    sce_rmse = mean_squared_error(y_test, sce_preds, squared=False)
    sce_r2 = r2_score(y_test, sce_preds)
    print(f"RMSE: {sce_rmse:.2f}")
    print(f"R²:   {sce_r2:.4f}")
    
    # Compare
    print("\n--- Improvement ---")
    rmse_improvement = (baseline_rmse - sce_rmse) / baseline_rmse * 100
    r2_improvement = (sce_r2 - baseline_r2) / (1 - baseline_r2) * 100 if baseline_r2 < 1 else 0
    print(f"RMSE improvement: {rmse_improvement:.2f}%")
    print(f"R² improvement:   {r2_improvement:.2f}%")


if __name__ == "__main__":
    main()
