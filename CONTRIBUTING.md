# Contributing to SCE

Thank you for your interest in contributing to Statistical Context Engineering!

## Development Setup

### Prerequisites

- Python 3.9+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/joint-hubs/sce.git
cd sce

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install with development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sce --cov-report=html

# Run specific test file
pytest tests/test_engine.py

# Run specific test
pytest tests/test_engine.py::test_cross_fitting_excludes_self_from_mean
```

### Code Quality

```bash
# Format code
ruff format sce tests

# Lint code
ruff check sce tests

# Type checking
mypy sce
```

---

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

### Key Guidelines

1. **Type hints**: All public functions must have type annotations
2. **Docstrings**: Use Google-style docstrings
3. **Line length**: 100 characters max
4. **Imports**: Use absolute imports, sorted by ruff

### Example

```python
def compute_aggregations(
    df: pd.DataFrame,
    target_col: str,
    group_cols: list[str],
    aggregations: list[AggregationMethod],
) -> pd.DataFrame:
    """Compute statistical aggregations for hierarchical groups.
    
    Args:
        df: Input DataFrame with target and group columns.
        target_col: Name of the target column to aggregate.
        group_cols: List of categorical columns defining the hierarchy.
        aggregations: List of aggregation methods to compute.
    
    Returns:
        DataFrame with computed aggregations for each group.
    
    Raises:
        ValueError: If target_col is not in df.
    """
    ...
```

---

## Module Metadata

All modules should include a metadata header:

```python
"""
@module: sce.engine
@status: stable
@depends: sce.config, sce.stats
@exports: StatisticalContextEngine
@paper_ref: Algorithm 1, Equations 3.1-3.4
"""
```

Status values: `stable`, `experimental`, `deprecated`

---

## Testing Guidelines

### Test Structure

```
tests/
├── conftest.py           # Shared fixtures
├── test_engine.py        # Engine tests
├── test_config.py        # Configuration tests
├── test_stats.py         # Statistics tests
└── test_pipeline.py      # Pipeline integration tests
```

### Writing Tests

1. One test per behavior
2. Use descriptive test names
3. Include edge cases
4. Use fixtures from `conftest.py`

```python
def test_cross_fitting_excludes_self_from_mean():
    """Verify that cross-fitting computes out-of-fold statistics."""
    # Arrange
    df = create_test_dataframe()
    config = ContextConfig(target_col="y", use_cross_fitting=True)
    engine = StatisticalContextEngine(config)
    
    # Act
    result = engine.fit_transform(df)
    
    # Assert
    # Each observation's context should NOT include itself
    assert not has_self_leakage(result)
```

---

## Pull Request Process

### Before Submitting

1. [ ] Code passes all tests: `pytest`
2. [ ] Code is formatted: `ruff format sce tests`
3. [ ] Code passes linting: `ruff check sce tests`
4. [ ] Type hints are complete: `mypy sce`
5. [ ] New features have tests
6. [ ] Documentation is updated

### PR Guidelines

1. **One feature per PR**: Keep changes focused
2. **Descriptive title**: Start with a verb (Add, Fix, Update, Remove)
3. **Link issues**: Reference related issues with `Fixes #123`
4. **Update changelog**: Add entry to CHANGELOG.md

### Review Process

1. All PRs require at least one review
2. CI must pass (tests, linting, type checking)
3. Maintainers may request changes
4. Squash merge preferred for clean history

---

## Reporting Issues

### Bug Reports

Include:
- Python version
- SCE version
- Minimal reproducible example
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches considered

---

## Questions?

- Open a [Discussion](https://github.com/joint-hubs/sce/discussions)
- Check existing issues first

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (CC BY-NC 4.0 for non-commercial use).
