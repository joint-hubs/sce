# Installation

## From PyPI (Recommended)

```bash
pip install stat-context
```

## With Optional Dependencies

```bash
# For development
pip install stat-context[dev]

# For dataset downloads
pip install stat-context[data]

# For visualization
pip install stat-context[viz]

# Everything
pip install stat-context[all]
```

## From Source

```bash
git clone https://github.com/joint-hubs/sce.git
cd sce
pip install -e .
```

## Requirements

- Python 3.9 or higher
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0

## Verify Installation

```python
import sce
print(sce.__version__)
```
