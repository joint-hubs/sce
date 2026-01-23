# Quick Start

This guide shows you how to use SCE to enrich your dataset with statistical context features.

## Basic Usage

```python
import pandas as pd
from sce import StatisticalContextEngine, ContextConfig

# Load your data
df = pd.read_csv("your_data.csv")

# Configure the engine
config = ContextConfig(
    target_col="price",           # Your target column
    use_cross_fitting=True,       # Prevents target leakage
    n_folds=5                     # Number of cross-validation folds
)

# Create and fit the engine
engine = StatisticalContextEngine(config)
enriched_df = engine.fit_transform(df)

# Check new features
new_cols = [c for c in enriched_df.columns if c not in df.columns]
print(f"Added {len(new_cols)} context features")
```

## With Manual Column Selection

```python
config = ContextConfig(
    target_col="price",
    categorical_cols=["city", "neighborhood", "property_type"],
    use_cross_fitting=True
)

engine = StatisticalContextEngine(config)
enriched_df = engine.fit_transform(df)
```

## In a scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

pipeline = Pipeline([
    ("context", StatisticalContextEngine(config)),
    ("model", GradientBoostingRegressor())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Full Experiment Pipeline

For complete experiments with train/test splits and evaluation:

```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sce import StatisticalContextEngine, ContextConfig

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=["price"]), df["price"], test_size=0.2
)

# Add target back for SCE (needed for cross-fitting)
train_df = X_train.copy()
train_df["price"] = y_train

# Fit SCE and transform
config = ContextConfig(target_col="price", use_cross_fitting=True)
engine = StatisticalContextEngine(config)
enriched_train = engine.fit_transform(train_df)
enriched_test = engine.transform(X_test)

# Train model
model = XGBRegressor(n_estimators=100)
model.fit(enriched_train.drop(columns=["price"]), y_train)

# Evaluate
preds = model.predict(enriched_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"SCE RMSE: {rmse:.2f}")
```

## What Features Are Created?

For each categorical column, SCE creates features with the pattern `{column}_{target}_{statistic}`:

| Feature | Description |
|---------|-------------|
| `{col}_{target}_mean` | Mean of target within group |
| `{col}_{target}_std` | Standard deviation within group |
| `{col}_{target}_median` | Median of target within group |
| `{col}_{target}_count` | Number of samples in group |
| `{col}_{target}_mean_fold_std` | Cross-fold variance (uncertainty) |

Example: For `city` column and `price` target → `city_price_mean`, `city_price_std`, etc.

## Next Steps

- See the [API Reference](../api/index.md) for detailed documentation
- Check [Experiments](../experiments.md) for benchmark results
