# Testing Workflow for ML Pipelines

> *"A test that doesn't teach you anything when it fails is a waste of electricity."*

---

## Why ML Testing Is Different

Web apps have clear inputs and outputs. ML pipelines have:
- **Stochastic behavior** (random sampling, model initialization)
- **Data dependencies** (tests need realistic fixtures)
- **Slow operations** (training a model takes minutes, not milliseconds)
- **Statistical properties** (outputs are distributions, not exact values)

This guide addresses each challenge.

---

## The Testing Pyramid for ML

```
                    ╱╲
                   ╱  ╲
                  ╱ E2E╲           1-2 smoke tests
                 ╱──────╲          (full pipeline runs)
                ╱        ╲
               ╱Integration╲       10-20 integration tests
              ╱────────────╲       (stage connections work)
             ╱              ╲
            ╱   Unit Tests   ╲     50-100 unit tests
           ╱──────────────────╲    (individual functions)
          ╱                    ╲
         ╱   Property Tests     ╲  20-30 property tests
        ╱────────────────────────╲ (statistical guarantees)
```

---

## Test Categories

### 1. Unit Tests — The Foundation

Test individual functions in isolation.

```python
# tests/unit/test_statistical_context.py

import pytest
import pandas as pd
from statistical_context import compute_group_statistics

class TestGroupStatistics:
    """
    The compute_group_statistics function is the workhorse of SCE.
    We need to verify it handles the full range of inputs correctly.
    """
    
    @pytest.fixture
    def simple_df(self):
        """Minimal dataset that exercises all code paths."""
        return pd.DataFrame({
            "city": ["NYC", "NYC", "LA", "LA", "LA"],
            "price": [100, 200, 150, 250, 350],
        })
    
    def test_returns_expected_statistics(self, simple_df):
        """Each grouping should produce exactly 9 statistics."""
        stats = compute_group_statistics(simple_df, groupby=["city"])
        
        expected_stats = {"min", "q25", "median", "mean", "q75", "max", 
                          "sd", "sd_relative", "mean_relative"}
        
        for city in ["NYC", "LA"]:
            assert set(stats[city].keys()) == expected_stats
    
    def test_median_is_middle_value(self, simple_df):
        """Sanity check: median should be the actual middle value."""
        stats = compute_group_statistics(simple_df, groupby=["city"])
        
        # NYC has [100, 200] → median = 150
        assert stats["NYC"]["median"] == 150.0
        
        # LA has [150, 250, 350] → median = 250
        assert stats["LA"]["median"] == 250.0
    
    def test_handles_single_value_groups(self):
        """Edge case: groups with one observation."""
        df = pd.DataFrame({"city": ["NYC"], "price": [100]})
        stats = compute_group_statistics(df, groupby=["city"])
        
        # SD should be 0 or NaN, not an error
        assert stats["NYC"]["sd"] == 0.0 or pd.isna(stats["NYC"]["sd"])
    
    def test_handles_empty_dataframe(self):
        """Edge case: empty input should fail gracefully."""
        df = pd.DataFrame({"city": [], "price": []})
        
        with pytest.raises(ValueError, match="empty"):
            compute_group_statistics(df, groupby=["city"])
```

### 2. Property Tests — Statistical Guarantees

Test invariants that should always hold, regardless of input.

```python
# tests/property/test_statistical_properties.py

import hypothesis
from hypothesis import given, strategies as st
import numpy as np
import pandas as pd
from statistical_context import compute_group_statistics

class TestStatisticalProperties:
    """
    These tests verify mathematical properties, not specific values.
    If any of these fail, something is fundamentally broken.
    """
    
    @given(prices=st.lists(st.floats(min_value=0, max_value=1e6), min_size=3))
    def test_median_between_min_and_max(self, prices):
        """Median must always lie within the range of values."""
        df = pd.DataFrame({"group": ["A"] * len(prices), "price": prices})
        stats = compute_group_statistics(df, groupby=["group"])
        
        assert stats["A"]["min"] <= stats["A"]["median"] <= stats["A"]["max"]
    
    @given(prices=st.lists(st.floats(min_value=0, max_value=1e6), min_size=2))
    def test_standard_deviation_non_negative(self, prices):
        """Standard deviation can never be negative."""
        df = pd.DataFrame({"group": ["A"] * len(prices), "price": prices})
        stats = compute_group_statistics(df, groupby=["group"])
        
        assert stats["A"]["sd"] >= 0
    
    @given(prices=st.lists(st.floats(min_value=1, max_value=1e6), min_size=2))
    def test_quartile_ordering(self, prices):
        """Quartiles must be ordered: q25 ≤ median ≤ q75."""
        df = pd.DataFrame({"group": ["A"] * len(prices), "price": prices})
        stats = compute_group_statistics(df, groupby=["group"])
        
        assert stats["A"]["q25"] <= stats["A"]["median"]
        assert stats["A"]["median"] <= stats["A"]["q75"]
```

### 3. Integration Tests — Pipeline Stages Connect

Test that outputs from one stage work as inputs to the next.

```python
# tests/integration/test_pipeline_stages.py

import pytest
import pandas as pd
from pathlib import Path
from statistical_context import StatisticalContextEngine
from feature_selection import FeatureSelectionPipeline

class TestPipelineIntegration:
    """
    These tests verify that pipeline stages connect properly.
    The contract: output of stage N is valid input for stage N+1.
    """
    
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a minimal but realistic test dataset."""
        df = pd.DataFrame({
            "city": ["NYC"] * 50 + ["LA"] * 50,
            "property_type": ["apt"] * 30 + ["house"] * 70,
            "size_sqft": np.random.randint(500, 3000, 100),
            "price": np.random.randint(100000, 500000, 100),
        })
        path = tmp_path / "test_data.csv"
        df.to_csv(path, index=False)
        return path
    
    def test_sce_output_feeds_into_feature_selection(self, sample_dataset):
        """SCE-enriched data should work with feature selection."""
        # Load and enrich
        df = pd.read_csv(sample_dataset)
        engine = StatisticalContextEngine(hierarchy_levels=["city", "property_type"])
        enriched = engine.fit_transform(df)
        
        # Feature selection should accept this
        selector = FeatureSelectionPipeline(correlation_threshold=0.8)
        selected = selector.fit_transform(enriched, target="price")
        
        # Should have fewer features than input (selection happened)
        assert len(selected.columns) < len(enriched.columns)
        # Target should be preserved
        assert "price" in selected.columns
    
    def test_model_training_accepts_selected_features(self, sample_dataset):
        """Selected features should work with XGBoost training."""
        # Full pipeline through feature selection
        df = pd.read_csv(sample_dataset)
        engine = StatisticalContextEngine(hierarchy_levels=["city"])
        enriched = engine.fit_transform(df)
        
        selector = FeatureSelectionPipeline()
        X, y = selector.fit_transform(enriched, target="price", return_arrays=True)
        
        # XGBoost should train without error
        import xgboost as xgb
        model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        # Model should produce predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y)
        assert not np.any(np.isnan(predictions))
```

### 4. Regression Tests — Changes Don't Break Things

Lock in expected behavior so refactoring doesn't introduce bugs.

```python
# tests/regression/test_known_outputs.py

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

class TestKnownOutputs:
    """
    These tests compare against known-good outputs.
    If they fail, either the code broke or the expected output needs updating.
    
    To regenerate expected outputs:
        pytest --regenerate-expected tests/regression/
    """
    
    @pytest.fixture
    def fixtures_dir(self):
        return Path(__file__).parent / "fixtures"
    
    def test_california_housing_baseline_rmse(self, fixtures_dir):
        """The California Housing baseline should produce consistent RMSE."""
        # This is our "golden" result from a known-good run
        expected_rmse = 0.523  # ± 0.01 tolerance for floating point
        
        # Run the actual pipeline
        from examples.quick_start import run_california_demo
        result = run_california_demo(seed=42)
        
        assert abs(result["rmse"] - expected_rmse) < 0.01, (
            f"RMSE changed from {expected_rmse} to {result['rmse']}. "
            f"If this is intentional, update the expected value."
        )
    
    def test_feature_importance_ranking_stable(self, fixtures_dir):
        """Top 5 features should be consistent across runs."""
        expected_top_5 = ["size_sqft", "city_median", "lat", "lon", "property_type_mean"]
        
        from examples.quick_start import run_california_demo
        result = run_california_demo(seed=42)
        actual_top_5 = result["feature_importance"].head(5).index.tolist()
        
        # Order might vary slightly, but the set should be the same
        assert set(actual_top_5) == set(expected_top_5), (
            f"Top features changed from {expected_top_5} to {actual_top_5}"
        )
```

### 5. Smoke Tests — It Runs End-to-End

One comprehensive test that runs the full pipeline.

```python
# tests/smoke/test_full_pipeline.py

import pytest
import subprocess
import sys
from pathlib import Path

class TestFullPipeline:
    """
    Smoke test: does the whole thing run without exploding?
    
    This is slow (minutes) but catches integration issues that
    unit tests miss. Run only in CI or before releases.
    """
    
    @pytest.mark.slow
    def test_sce_analysis_script_runs(self, tmp_path):
        """The main CLI script should complete without errors."""
        result = subprocess.run(
            [
                sys.executable, "scripts/sce_analysis.py",
                "--dataset", "sample",  # Use a small test dataset
                "--output-dir", str(tmp_path),
                "--sampling-pct", "1",  # Minimal sampling for speed
            ],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )
        
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"
        
        # Verify outputs exist
        assert (tmp_path / "report.md").exists()
        assert (tmp_path / "data" / "metadata.json").exists()
        assert (tmp_path / "figures").is_dir()
```

---

## Test Fixtures Strategy

### The Fixture Hierarchy

```
tests/fixtures/
├── minimal/          # Smallest possible valid inputs (5-10 rows)
├── realistic/        # Representative data (100-1000 rows)
├── edge_cases/       # Known problematic inputs
│   ├── missing_values.csv
│   ├── single_category.csv
│   ├── all_same_value.csv
│   └── extreme_outliers.csv
└── golden/           # Expected outputs for regression tests
```

### Creating Good Fixtures

```python
# conftest.py

import pytest
import pandas as pd
import numpy as np

@pytest.fixture(scope="session")
def realistic_listings():
    """
    A realistic but synthetic dataset for testing.
    
    Why synthetic: Real data has privacy concerns and licensing issues.
    Why realistic: Tests should exercise real-world patterns.
    """
    np.random.seed(42)
    n = 1000
    
    cities = np.random.choice(["NYC", "LA", "Chicago", "Miami"], n, 
                               p=[0.3, 0.3, 0.2, 0.2])
    
    # Price correlates with city (NYC most expensive)
    base_prices = {"NYC": 500000, "LA": 400000, "Chicago": 250000, "Miami": 350000}
    prices = [base_prices[c] + np.random.normal(0, 50000) for c in cities]
    
    return pd.DataFrame({
        "city": cities,
        "price": prices,
        "size_sqft": np.random.randint(500, 5000, n),
        "bedrooms": np.random.randint(1, 6, n),
    })

@pytest.fixture
def edge_case_missing_target():
    """Dataset where some target values are NaN."""
    return pd.DataFrame({
        "city": ["NYC", "LA", "NYC"],
        "price": [100, np.nan, 200],
    })
```

---

## Running Tests

### Development Workflow

```bash
# Run fast tests only (excludes @pytest.mark.slow)
pytest tests/ -v -m "not slow"

# Run specific test file
pytest tests/unit/test_statistical_context.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run property tests with more examples
pytest tests/property/ --hypothesis-seed=42 -v
```

### CI Workflow

```yaml
# .github/workflows/test.yml

name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
      
      - name: Run tests
        run: pytest tests/ -v --tb=short
      
      - name: Run slow tests (smoke)
        run: pytest tests/smoke/ -v -m slow --tb=short
```

---

## Testing Checklist

### Before Committing

- [ ] All unit tests pass locally
- [ ] New code has corresponding tests
- [ ] Edge cases are covered
- [ ] Tests document expected behavior (good names, docstrings)

### Before Release

- [ ] Full test suite passes (including slow tests)
- [ ] Coverage report reviewed
- [ ] Regression tests verify expected outputs
- [ ] Smoke test completes successfully

---

## Common Pitfalls

### 1. Testing the Framework, Not Your Code

```python
# BAD: Tests pandas, not your code
def test_dataframe_merge():
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df1.merge(df2, on="a")
    assert len(result) == 2

# GOOD: Tests your business logic
def test_context_merge_preserves_all_listings():
    listings = create_test_listings(100)
    context = compute_context(listings)
    enriched = merge_context(listings, context)
    
    assert len(enriched) == len(listings), "Merge dropped rows"
```

### 2. Non-Deterministic Tests

```python
# BAD: Fails randomly
def test_model_accuracy():
    model = train_model(X, y)
    assert model.score(X_test, y_test) > 0.8  # Sometimes fails!

# GOOD: Deterministic
def test_model_accuracy():
    model = train_model(X, y, random_state=42)
    score = model.score(X_test, y_test)
    assert abs(score - 0.847) < 0.01  # Expected value ± tolerance
```

### 3. Tests That Take Forever

```python
# BAD: Uses full dataset
def test_feature_engineering():
    df = load_full_dataset()  # 10 million rows
    result = engineer_features(df)  # Takes 5 minutes

# GOOD: Uses minimal fixture
def test_feature_engineering(minimal_fixture):
    result = engineer_features(minimal_fixture)  # 10 rows, instant
    assert_expected_columns(result)
```

---

*"Tests are not a tax you pay to the CI gods. Tests are documentation that runs itself."*
