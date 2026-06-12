# SCE Configuration Files

This directory contains TOML configuration files for SCE experiments.

## Minimal Configuration (v0.3.0+)

With auto-detection, configs are now simpler:

```toml
[dataset]
name = "my_dataset"
path = "data/parquet/my_data.parquet"

[target]
column = "price"

[sce]
aggregations = ["mean", "median", "std", "count"]
min_group_size = 3
use_cross_fitting = true
n_folds = 5
```

**That's it!** Categorical columns are auto-detected from the DataFrame.

## Remote Source Notes

Dataset configs still point to the local parquet path. Remote acquisition is declared in the manifest entry for that parquet file.

Supported manifest source formats:

- `https://...` for direct file downloads
- `kaggle://datasets/<owner>/<dataset>/<file>` for Kaggle datasets
- `kaggle://competitions/<competition>/<file>` for Kaggle competitions

Remote sources should be converted into the deterministic parquet files used by experiments.

## Optional Overrides

```toml
[sce]
# Override auto-detection
categorical_cols = ["city", "room_type", "property_type"]

# Require at least this many categorical columns to run SCE
min_categorical_columns = 1

# Control interaction explosion
include_interactions = true
max_interaction_depth = 2

# Control categorical detection size
max_cardinality = 100

# Control detection threshold
max_cardinality = 100  # Columns with more unique values are skipped

# Feature generation
include_interactions = true  # 2-way categorical interactions
max_interaction_depth = 2    # Only pairs (A×B), not triples

[run.feature_pruning]
# Drop columns with too many missing values or zero variance
missing_threshold = 0.2
drop_zero_variance = true
```

## Available Aggregations

| Name | Description |
|------|-------------|
| `mean` | Arithmetic mean |
| `median` | 50th percentile |
| `std` | Standard deviation |
| `count` | Group size |
| `q05` - `q95` | Quantiles (5%, 10%, 20%, 25%, 33%, 66%, 75%, 80%, 90%, 95%) |
| `min`, `max` | Range bounds |
| `var` | Variance |
| `cv` | Coefficient of variation |
| `iqr` | Interquartile range |

## Available Datasets

| Config File | Dataset | Target |
|-------------|---------|--------|
| `rental_poland_short.toml` | Short-term rentals (Airbnb-style) | `price_PLN_per_night` |
| `melbourne_housing.toml` | Melbourne housing prices | `Price` |
| `m5_store_dept_daily.toml` | Hierarchical daily demand (generated from M5 raw files) | `demand` |
| `walmart_weekly.toml` | Walmart weekly store-department sales | `Weekly_Sales` |
| `rossmann_daily.toml` | Rossmann daily store sales | `Sales` |

Configs under `experimental/` are excluded from `list_datasets()` and
`run.py --all` — they did not pass the leakage diagnostics gate
(permuted-target / shuffled-groups) and need investigation or retuning
before their results can be reported. See `experimental/README.md`.

## Migration from v0.2.x

The `[hierarchy]` section is deprecated. Remove it:

```diff
  [dataset]
  name = "my_dataset"
  
- [hierarchy]
- levels = ["city", "room_type", "property_type"]
  
  [target]
  column = "price"
```

Categorical columns are now auto-detected, or you can specify them in `[sce].categorical_cols`.
