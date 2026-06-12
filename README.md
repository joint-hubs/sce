# Statistical Context Engineering (SCE)

**Leakage-safe hierarchical target statistics for tabular regression.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Tests](https://github.com/joint-hubs/sce/actions/workflows/ci.yml/badge.svg)](https://github.com/joint-hubs/sce/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/stat-context.svg)](https://pypi.org/project/stat-context/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://joint-hubs.github.io/sce)

SCE enriches a tabular dataset with group-level statistics of the target
variable (mean, quantiles, spread, counts) computed over categorical columns —
the "context" each row lives in. The hard part of target encoding is doing it
**without leaking the target**: SCE uses out-of-fold cross-fitting on the
training set, temporal-aware folds for time series, and ships a diagnostics
suite that *proves* the absence of leakage on your data.

```
Input:    [city, rooms, sqft]                            → RMSE 1000
Enriched: [city, rooms, sqft, city_price_mean,
           city_price_q20, city_price_std, ...]          → RMSE  900
```

**Why this instead of plain target encoding?**

- **Cross-fitted, not naive** — training rows only ever see statistics computed
  from *other* folds, so the model cannot memorize its own target.
- **Temporal mode** — for time-series data, statistics come strictly from the
  past (rolling folds), never from the future.
- **Rich context, not just the mean** — quantiles, spread, and group size let
  the model see the whole distribution a row belongs to.
- **Verified** — permuted-target and shuffled-groups diagnostics confirm the
  enrichment adds nothing on noise (a leaking encoder *would* improve on noise).

---

## Installation

```bash
pip install stat-context
```

```python
import sce  # package name on PyPI is stat-context, import name is sce
```

Optional extras:

```bash
pip install stat-context[models]  # LightGBM + CatBoost backends
pip install stat-context[data]    # remote dataset download helpers
pip install stat-context[dev]     # tests, linters
```

---

## 60-second quickstart

Copy-paste runnable — synthetic data, no files needed:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor

from sce import StatisticalContextEngine, ContextConfig, AggregationMethod

# --- toy data: price depends on a hidden per-city level the model can't see
rng = np.random.default_rng(42)
n = 8000
cities = rng.choice([f"city_{i}" for i in range(40)], size=n)
city_level = {f"city_{i}": rng.normal(100, 30) for i in range(40)}
df = pd.DataFrame({
    "city": cities,
    "sqft": rng.uniform(20, 120, size=n),
    "rooms": rng.integers(1, 6, size=n),
})
df["price"] = (
    df["city"].map(city_level) * 10 + df["sqft"] * 50 + rng.normal(0, 300, size=n)
)

# --- split FIRST, then enrich (this is the leakage-safe order)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

config = ContextConfig(
    target_col="price",
    categorical_cols=["city"],
    aggregations=[AggregationMethod.MEAN, AggregationMethod.STD, AggregationMethod.COUNT],
    use_cross_fitting=True,   # train rows get out-of-fold stats
    n_folds=5,
)
engine = StatisticalContextEngine(config)
train_enriched = engine.fit_transform(train_df)                       # OOF stats
test_enriched = engine.transform(test_df.drop(columns=["price"]))     # full-train stats

# --- compare baseline vs enriched
features_base = ["sqft", "rooms"]
features_sce = features_base + [c for c in train_enriched.columns
                                if c.startswith("city_price_")]

y_train, y_test = train_enriched["price"], test_df["price"]
for name, feats in [("baseline", features_base), ("with SCE", features_sce)]:
    model = XGBRegressor(n_estimators=200, random_state=42)
    model.fit(train_enriched[feats], y_train)
    rmse = root_mean_squared_error(y_test, model.predict(test_enriched[feats]))
    print(f"{name:9s} RMSE: {rmse:,.0f}")
```

Output — the enriched model recovers the hidden city level:

```
baseline  RMSE: 475
with SCE  RMSE: 362
```

New features follow the pattern `{column}_{target}_{stat}`, e.g.
`city_price_mean`, `city_price_std`, `city_price_count`, plus fold-variance
features (`*_fold_std`, `*_fold_lower`, `*_fold_upper`) that quantify how
stable each statistic is across folds. To see what was added, diff the
DataFrame columns before and after enrichment.

### Time-series data

For temporal data, switch to rolling cross-fitting so statistics never look
into the future:

```python
config = ContextConfig(
    target_col="sales",
    categorical_cols=["store", "dept"],
    use_cross_fitting=True,
    cross_fit_strategy="rolling",   # monotonic folds: train_max < val_min
    time_col="date",
    n_folds=5,
)
```

---

## Benchmark results (report-grade)

Every number below comes from a run that passed the full diagnostics gate:
clean git tree, full dataset (no subsampling), and all leakage diagnostics
green — including **permuted-target** (SCE must show ~zero improvement when
targets are shuffled; a leaking pipeline improves even on noise).

| Dataset | Rows | Split | Baseline → SCE RMSE | Improvement |
|---------|-----:|-------|---------------------|------------:|
| `rental_poland_short` (Airbnb) | 1,185 | random | — | **+10.97%** |
| `rossmann_daily` (store sales) | 844,338 | temporal | 950 → 855 | **+9.90%** |
| `walmart_weekly` (dept sales) | 420,212 | temporal | 6,769 → 6,339 | **+6.35%** |
| `melbourne_housing` (prices) | 27,247 | random | — | **+2.19%** |
| `m5_store_dept_daily` (demand) | 23,529 | temporal | — | **+1.14%** |

Model: XGBoost with identical hyperparameters for baseline and SCE; the only
difference is the added context features. Honest caveat: gains depend on how
informative the categorical hierarchy is — datasets where SCE shows no
significant advantage are excluded from the benchmark set (see
`configs/experimental/`) rather than reported with inflated numbers.

---

## Reproducing the benchmarks

Requires a repo clone (datasets are rebuilt locally — Kaggle sources cannot be
redistributed):

```bash
git clone https://github.com/joint-hubs/sce.git && cd sce
pip install -e .[dev,data]

# rebuild the Kaggle-derived datasets (needs Kaggle API credentials)
python scripts/prepare_new_datasets.py
python scripts/prepare_m5_dataset.py --download

sce datasets list                                  # see what is available
python scripts/run.py --dataset rental_poland_short   # single experiment
python scripts/run.py --all                        # full benchmark
```

Each run writes a results directory with `metadata.json` (git SHA, config
hash, seed, metrics, diagnostics, promotion status) — every reported number
is traceable to an exact commit and config.

### Leakage diagnostics

```bash
python -m scripts.diagnostics.permuted_target  --dataset rossmann_daily
python -m scripts.diagnostics.shuffled_groups  --dataset rossmann_daily
python -m scripts.diagnostics.crossfit_ab      --dataset rossmann_daily
```

| Diagnostic | Question it answers | Pass criterion |
|---|---|---|
| `permuted_target` | Does SCE "improve" on pure noise? | advantage < 1% on shuffled targets |
| `shuffled_groups` | Does SCE need real group structure? | no advantage with shuffled categories |
| `crossfit_ab` | What does cross-fitting cost/protect? | informational |
| feature dominance | Is one feature doing everything? | top-3 share < 70% |

---

## Configuration reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `target_col` | `str` | required | Target variable column |
| `categorical_cols` | `list[str]` | `None` | Grouping columns (auto-detected if `None`) |
| `aggregations` | `list[AggregationMethod]` | `[MEAN, MEDIAN, STD, Q05, Q20, Q80, Q95, COUNT]` | Statistics to compute |
| `use_cross_fitting` | `bool` | `True` | Out-of-fold statistics on train |
| `cross_fit_strategy` | `str` | `"random"` | `"random"` or `"rolling"` (temporal) |
| `time_col` | `str` | `None` | Required for `"rolling"` |
| `n_folds` | `int` | `5` | Cross-fitting folds |
| `min_group_size` | `int` | `5` | Groups smaller than this back off to global stats |
| `include_global_stats` | `bool` | `True` | Dataset-wide statistics |
| `include_interactions` | `bool` | `True` | Cross-column group hierarchies (`city×type`) |

```python
class StatisticalContextEngine(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self: ...
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame: ...
```

Scikit-learn compatible — drops into a `Pipeline`. Full API docs:
[joint-hubs.github.io/sce](https://joint-hubs.github.io/sce/api/)

---

## How it works

1. **Grouping** — categorical columns (and optionally their interactions)
   define groups: `city`, `city×property_type`, …
2. **Aggregation** — for each group, compute the configured target statistics.
3. **Cross-fitting** — on the training set, each row receives statistics
   computed *without its own fold* (random K-fold, or rolling time-ordered
   folds for temporal data). Small groups back off to global statistics.
4. **Inference** — test/production rows receive statistics computed from the
   full training set. The target is never needed at transform time.

The evaluation protocol used for all reported numbers is split-first:
train/test split happens **before** any enrichment, encoders and pruning are
fit on train only, and the test set is touched exactly once.

---

## Project layout

```
sce/                  # the library (pip install stat-context)
├── engine.py         #   StatisticalContextEngine
├── config.py         #   ContextConfig, AggregationMethod
├── models.py         #   model factory (xgboost core; lightgbm/catboost optional)
├── stats.py          #   aggregation functions
└── io/               #   dataset registry + loading
configs/              # benchmark dataset configs (TOML)
configs/experimental/ # configs excluded from the benchmark set
scripts/              # experiment runner, diagnostics, dataset preparation
tests/                # pytest suite (incl. leakage guard tests)
```

---

## License

This software is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) — free for academic, research, and commercial use.

The accompanying research paper remains under CC BY-NC 4.0.

---

## Citation

```bibtex
@inproceedings{sce2026,
  title={Statistical Context Engineering for Hierarchical Tabular Data},
  author={Stachowicz, Mateusz and Halkiewicz, Stanis{\l}aw},
  booktitle={Proceedings of the International Conference on Machine Learning},
  year={2026}
}
```

---

## Links

- [Source code](https://github.com/joint-hubs/sce)
- [Issue tracker](https://github.com/joint-hubs/sce/issues)
- [Documentation](https://joint-hubs.github.io/sce/)
- [Contributing](https://github.com/joint-hubs/sce/blob/main/CONTRIBUTING.md)
