# Statistical Context Engineering (SCE)

> **Hierarchical Feature Enrichment for Regression Models**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Tests](https://github.com/joint-hubs/sce/actions/workflows/ci.yml/badge.svg)](https://github.com/joint-hubs/sce/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/joint-hubs/sce/branch/main/graph/badge.svg)](https://codecov.io/gh/joint-hubs/sce)

---

## What is SCE?

**Statistical Context Engineering** enriches regression datasets with hierarchical statistical features. By aggregating target statistics at multiple levels of granularity (e.g., country → region → city → neighborhood), SCE provides models with powerful contextual information while preventing data leakage through cross-fitting.

### Key Features

- 🎯 **Leakage-Safe**: Out-of-fold aggregation prevents test set contamination
- 🔍 **Auto-Detection**: Categorical columns detected automatically from your DataFrame
- 🏗️ **Hierarchical**: Captures context at multiple granularity levels + interactions
- 🔌 **Scikit-learn Compatible**: Drop-in transformer for existing pipelines
- 📊 **Theory-Backed**: Grounded in Bayesian estimation and cooperative game theory
- ⚡ **Production-Ready**: Typed, tested, and documented

---

## Installation

### From PyPI (Recommended)

```bash
pip install sce
```

### From Source

```bash
git clone https://github.com/joint-hubs/sce.git
cd sce
pip install -e .
```

### With Optional Dependencies

```bash
pip install sce[dev]   # Development tools
pip install sce[data]  # Remote dataset fetching
pip install sce[viz]   # Visualization tools
pip install sce[all]   # Everything
```

---

## Quick Start

```python
from sce import StatisticalContextEngine, ContextConfig, AggregationMethod
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your data
df = pd.read_csv("rental_data.csv")

# Configure SCE (categorical columns auto-detected!)
config = ContextConfig(
    target_col="price",
    aggregations=[AggregationMethod.MEAN, AggregationMethod.STD],
    use_cross_fitting=True,  # Prevents leakage
    n_folds=5
)

# Create pipeline
pipeline = Pipeline([
    ("sce", StatisticalContextEngine(config)),
    ("model", XGBRegressor(n_estimators=100))
])

# Train
X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
```

### Manual Column Specification (Optional)

```python
config = ContextConfig(
    target_col="price",
    categorical_cols=["country", "region", "city", "neighborhood"],
    aggregations=[AggregationMethod.MEAN, AggregationMethod.STD],
    use_cross_fitting=True
)
```

---

## How It Works

SCE computes statistical summaries of the target variable within hierarchical groups:

1. **Define Hierarchies**: Categorical columns form a natural hierarchy (e.g., `city` → `neighborhood`)
2. **Compute Statistics**: For each group, compute mean, std, median, quantiles, etc.
3. **Cross-Fitting**: Use out-of-fold aggregation to prevent leakage
4. **Enrich Features**: Append context features to original dataset

```
Original: [city, beds, sqft]
Enriched: [city, beds, sqft, city_mean, city_std, city_count, ...]
```

---

## Datasets

Four real-world datasets are included for benchmarking:

| Dataset | Domain | Samples | Hierarchy Levels | Location |
|---------|--------|---------|------------------|----------|
| `rental_poland_short` | Short-term rentals | 12,847 | 4 | In-repo |
| `rental_poland_long` | Long-term rentals | 28,391 | 4 | In-repo |
| `rental_uae_contracts` | Dubai rental contracts | 156,203 | 3 | Remote |
| `sales_uae_transactions` | Dubai property sales | 89,456 | 3 | Remote |

Remote datasets download automatically on first use.

```python
from sce.io import load_dataset

df = load_dataset("rental_poland_short")  # Local
df = load_dataset("rental_uae_contracts")  # Auto-downloads
```

---

## Reproducing Paper Results

```bash
# Run all experiments
python scripts/run.py --all

# Run specific dataset
python scripts/run.py --dataset rental_uae_contracts

# Run comprehensive search
python scripts/run.py --dataset rental_uae_contracts --search

# Generate paper figures
python scripts/generate_paper_appendix_figures.py
```

Results are saved to `results/` with metrics, visualizations, and detailed reports.

---

## Configuration Reference

```python
from sce import ContextConfig, AggregationMethod

config = ContextConfig(
    # Required
    target_col="price",
    
    # Categorical detection (auto if not specified)
    categorical_cols=None,  # Auto-detect from DataFrame
    
    # Aggregations to compute
    aggregations=[
        AggregationMethod.MEAN,
        AggregationMethod.MEDIAN,
        AggregationMethod.STD,
        AggregationMethod.Q25,
        AggregationMethod.Q75,
        AggregationMethod.COUNT
    ],
    
    # Leakage prevention
    use_cross_fitting=True,  # Always True for training
    n_folds=5,
    
    # Advanced
    min_group_size=5,           # Minimum samples per group
    include_global_stats=True,  # Dataset-wide statistics
    include_interactions=False, # Cross-column hierarchies
)
```

---

## Architecture

```
sce/
├── engine.py       # StatisticalContextEngine (core transformer)
├── config.py       # Configuration dataclasses
├── stats.py        # Statistical aggregators
├── pipeline.py     # Scikit-learn pipeline utilities
├── cleanup.py      # Feature cleanup and NaN handling
├── importance.py   # Feature importance analysis
└── io/             # Dataset loading utilities
```

---

## Citation

If you use SCE in your research, please cite:

```bibtex
@inproceedings{sce2025,
  title={Statistical Context Engineering: Hierarchical Feature Enrichment for Regression},
  author={Stachowicz, Mateusz and Halkiewicz, Stanisław},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```

---

## License

- **Academic/Research Use**: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (free)
- **Commercial Use**: Contact authors for licensing

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Links

- 📖 [Documentation](https://github.com/joint-hubs/sce#readme)
- 🐛 [Issue Tracker](https://github.com/joint-hubs/sce/issues)
- 💬 [Discussions](https://github.com/joint-hubs/sce/discussions)
