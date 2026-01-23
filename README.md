# Statistical Context Engineering (SCE)

**Hierarchical feature enrichment for regression with leakage-safe cross-fitting**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Tests](https://github.com/joint-hubs/sce/actions/workflows/ci.yml/badge.svg)](https://github.com/joint-hubs/sce/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/stat-context.svg)](https://pypi.org/project/stat-context/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://joint-hubs.github.io/sce)

---

## Overview

Statistical Context Engineering (SCE) is a Python library for enriching tabular regression datasets with hierarchical statistical features. The method computes target-variable aggregations at multiple levels of categorical granularity (e.g., region → city → neighborhood) while preventing information leakage through k-fold cross-fitting.

The approach is grounded in Bayesian estimation principles and cooperative game theory, providing a principled framework for incorporating group-level context into predictive models.

### Features

- **Leakage prevention**: Out-of-fold aggregation ensures no target information from the test set contaminates training features
- **Automatic detection**: Categorical columns are inferred from DataFrame dtypes when not explicitly specified
- **Hierarchical aggregation**: Captures statistical context at multiple granularity levels, including cross-column interactions
- **Scikit-learn compatibility**: Implements the transformer interface for seamless pipeline integration
- **Uncertainty quantification**: Cross-fitting provides fold-variance estimates for context features

---

## Installation

### From PyPI

```bash
pip install stat-context
```

### From Source

```bash
git clone https://github.com/joint-hubs/sce.git
cd sce
pip install -e .
```

### Optional Dependencies

```bash
pip install stat-context[dev]   # Development and testing tools
pip install stat-context[data]  # Remote dataset fetching
pip install stat-context[all]   # All optional dependencies
```

---

## Usage

### Basic Example

```python
from sce import StatisticalContextEngine, ContextConfig, AggregationMethod
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd

# Load data
df = pd.read_csv("rental_data.csv")
X = df.drop(columns=["price"])
y = df["price"]

# Configure the context engine
config = ContextConfig(
    target_col="price",
    aggregations=[AggregationMethod.MEAN, AggregationMethod.STD],
    use_cross_fitting=True,
    n_folds=5
)

# Build pipeline
pipeline = Pipeline([
    ("sce", StatisticalContextEngine(config)),
    ("model", XGBRegressor(n_estimators=100))
])

# Train and predict
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Explicit Column Specification

```python
config = ContextConfig(
    target_col="price",
    categorical_cols=["country", "region", "city", "neighborhood"],
    aggregations=[
        AggregationMethod.MEAN,
        AggregationMethod.MEDIAN,
        AggregationMethod.STD,
        AggregationMethod.Q25,
        AggregationMethod.Q75,
        AggregationMethod.COUNT
    ],
    use_cross_fitting=True,
    n_folds=5,
    min_group_size=5,
    include_global_stats=True,
    include_interactions=True
)
```

---

## Method

SCE operates in four stages:

1. **Hierarchy identification**: Categorical columns define grouping levels (e.g., `city`, `neighborhood`)
2. **Statistical aggregation**: For each group, compute configurable statistics (mean, std, quantiles, etc.)
3. **Cross-fitting**: K-fold out-of-fold aggregation prevents target leakage during training
4. **Feature enrichment**: Context features are appended to the original feature matrix

The transformation extends the feature space as follows:

```
Input:    [city, beds, sqft]
Output:   [city, beds, sqft, city_mean, city_std, city_count, ...]
```

During inference, aggregations are computed from the full training set without cross-fitting.

---

## Datasets

Four benchmark datasets are provided for reproducibility:

| Dataset | Domain | Samples | Hier. Cols | Base Feats | +SCE Feats |
|---------|--------|---------|------------|------------|------------|
| `rental_poland_short` | Short-term rentals (Airbnb) | 1,185 | 4 | 9 | 504 |
| `rental_poland_long` | Long-term rentals (Otodom) | 1,016 | 5 | 4 | 724 |
| `rental_uae_contracts` | Dubai rental contracts | 50,000 | 6 | 8 | 323 |
| `sales_uae_transactions` | Dubai property sales | 50,000 | 7 | 10 | 370 |

```python
from sce.io import load_dataset

df = load_dataset("rental_poland_short")   # Bundled with package
df = load_dataset("rental_uae_contracts")  # Downloaded on first use
```

---

## Experimental Results

Experiments compare baseline models (XGBoost with original features) against SCE-enriched models using comprehensive hyperparameter search.

| Dataset | Baseline RMSE | +SCE RMSE | RMSE Reduction | R² Change |
|---------|---------------|-----------|----------------|------------|
| Airbnb Poland | 27,368 | 22,541 | 17.6% | +24.49 pp |
| Dubai Rentals | 465,037 | 360,267 | 22.5% | +3.83 pp |
| Dubai Transactions | 32,489,660 | 26,353,228 | 18.9% | +25.83 pp |
| Otodom Poland | 4,581 | 4,541 | 0.9% | +1.55 pp |

Results vary by dataset characteristics. The method shows larger improvements on datasets with informative categorical hierarchies and sufficient group sizes. See the `results/` directory for detailed experimental outputs.

---

## Reproducing Experiments

```bash
# Run experiments on all datasets
python scripts/run.py --all --search

# Run on a specific dataset
python scripts/run.py --dataset rental_uae_contracts --search

# Generate figures and tables
python scripts/generate_figures.py
python scripts/generate_paper_appendix_figures.py
```

---

## API Reference

### ContextConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_col` | `str` | Required | Name of the target variable column |
| `categorical_cols` | `list[str]` | `None` | Columns to aggregate over (auto-detected if None) |
| `aggregations` | `list[AggregationMethod]` | `[MEAN, MEDIAN, STD, Q05, Q20, Q80, Q95, COUNT]` | Statistics to compute |
| `use_cross_fitting` | `bool` | `True` | Enable out-of-fold aggregation |
| `n_folds` | `int` | `5` | Number of cross-fitting folds |
| `min_group_size` | `int` | `5` | Minimum samples required per group |
| `include_global_stats` | `bool` | `True` | Include dataset-wide statistics |
| `include_interactions` | `bool` | `True` | Include cross-column hierarchies |

### StatisticalContextEngine

```python
class StatisticalContextEngine(BaseEstimator, TransformerMixin):
    def __init__(self, config: ContextConfig): ...
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self: ...
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame: ...
```

Full API reference: [Documentation](https://joint-hubs.github.io/sce/api/)

---

## Package Structure

```
sce/
├── engine.py       # StatisticalContextEngine transformer
├── config.py       # ContextConfig and AggregationMethod
├── stats.py        # Aggregation functions
├── pipeline.py     # Pipeline utilities
├── cleanup.py      # NaN handling and feature cleanup
├── importance.py   # Feature importance analysis
└── io/             # Dataset loading
```

---

## License

This software is released under the [Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/).

- **Academic and research use**: Permitted with attribution
- **Commercial use**: Requires a separate license agreement

---

## Citation

If you use this software in your research, please cite:

```bibtex
@inproceedings{sce2026,
  title={Statistical Context Engineering for Hierarchical Tabular Data},
  author={Stachowicz, Mateusz and Halkiewicz, Stanis{\l}aw},
  booktitle={Proceedings of the International Conference on Machine Learning},
  year={2026}
}
```

---

## Contributing

Contributions are welcome. Please see [CONTRIBUTING.md](https://github.com/joint-hubs/sce/blob/main/CONTRIBUTING.md) for guidelines on code style, testing, and pull request procedures.

---

## Links

- [Source Code](https://github.com/joint-hubs/sce)
- [Issue Tracker](https://github.com/joint-hubs/sce/issues)
- [Documentation](https://joint-hubs.github.io/sce/)
