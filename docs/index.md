# Statistical Context Engineering

[![PyPI version](https://badge.fury.io/py/stat-context.svg)](https://pypi.org/project/stat-context/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Statistical Context Engineering (SCE)** is a feature engineering methodology that enriches tabular datasets with hierarchical statistical context, improving regression model performance.

## Key Features

- **Leakage-safe**: Uses cross-fitting (out-of-fold aggregation) to prevent target leakage
- **Hierarchical**: Supports multi-level categorical hierarchies with automatic backoff
- **scikit-learn compatible**: Drop-in transformer for ML pipelines
- **Auto-detection**: Automatically identifies categorical columns

## Installation

```bash
pip install stat-context
```

## Quick Example

```python
from sce import StatisticalContextEngine, ContextConfig

# Configure the engine
config = ContextConfig(
    target_col="price",
    use_cross_fitting=True  # Prevents leakage
)

# Create and fit the engine
engine = StatisticalContextEngine(config)
enriched_df = engine.fit_transform(train_df)

# Now enriched_df has additional statistical context features
```

## How It Works

SCE computes group-level statistics (mean, std, count, etc.) for categorical columns, creating features that capture the "context" of each observation. The cross-fitting approach ensures these features don't leak target information.

For example, for a property in a specific neighborhood:
- `neighborhood_price_mean`: Average price in that neighborhood
- `neighborhood_price_std`: Price variation in that neighborhood  
- `neighborhood_price_count`: Number of samples in that neighborhood

## Documentation

- [Getting Started](getting-started/installation.md)
- [API Reference](api/index.md)
- [Experiments](experiments.md)

## Citation

If you use SCE in your research, please cite:

```bibtex
@software{stachowicz2025sce,
  author = {Stachowicz, Mateusz and Halkiewicz, Stanisław},
  title = {Statistical Context Engineering: Hierarchical Feature Enrichment for Regression Models},
  year = {2025},
  url = {https://github.com/joint-hubs/sce}
}
```
