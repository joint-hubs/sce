# Examples

This directory contains example scripts demonstrating SCE usage.

## Examples

### basic_usage.py

Demonstrates the core SCE workflow:
- Loading a dataset
- Configuring SCE with aggregations
- Enriching features with cross-fitting
- Training XGBoost models
- Comparing baseline vs SCE-enriched performance

```bash
python examples/basic_usage.py
```

### Expected Output

```
Loading dataset...
Dataset shape: (12847, 15)

Fitting SCE...
Original features: 14
Enriched features: 58
Context features added: 44

--- Baseline (no SCE) ---
RMSE: 245.32
R²:   0.7234

--- With SCE ---
RMSE: 198.45
R²:   0.8567

--- Improvement ---
RMSE improvement: 19.10%
R² improvement: 48.21%
```

## More Examples

Additional examples coming soon:
- Pipeline integration with scikit-learn
- Custom aggregation methods
- Large-scale dataset processing
- Feature importance analysis
