# Test configuration
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sce.config import AggregationMethod, ContextConfig

# Add sce to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_data():
    """Create sample hierarchical data for testing."""
    np.random.seed(42)
    n = 100

    cities = np.random.choice(["NYC", "LA", "CHI"], n)
    neighborhoods = []
    for city in cities:
        if city == "NYC":
            neighborhoods.append(np.random.choice(["Manhattan", "Brooklyn"], 1)[0])
        elif city == "LA":
            neighborhoods.append(np.random.choice(["Hollywood", "Venice"], 1)[0])
        else:
            neighborhoods.append(np.random.choice(["Loop", "Lincoln"], 1)[0])

    df = pd.DataFrame(
        {
            "city": cities,
            "neighborhood": neighborhoods,
            "price": np.random.randint(100, 500, n),
            "sqft": np.random.randint(500, 2000, n),
        }
    )

    return df


@pytest.fixture
def basic_config():
    """Basic SCE configuration for testing."""
    return ContextConfig(
        categorical_cols=["city", "neighborhood"],
        target_col="price",
        aggregations=[AggregationMethod.MEAN, AggregationMethod.STD, AggregationMethod.MEDIAN],
        min_group_size=3,
        use_cross_fitting=False,
        n_folds=3,
    )


@pytest.fixture
def cross_fit_config(basic_config):
    """Configuration with cross-fitting enabled."""
    basic_config.use_cross_fitting = True
    return basic_config
