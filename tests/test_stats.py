"""
Tests for statistical aggregators.
"""

import pytest
import pandas as pd
import numpy as np

from sce.stats import (
    StatsAggregator,
    compute_aggregations,
    compute_relative_features,
    apply_hierarchical_backoff,
    _get_level_names,
)
from sce.config import AggregationMethod


def test_aggregator_initialization():
    """Test StatsAggregator can be initialized."""
    methods = [AggregationMethod.MEAN, AggregationMethod.STD]
    agg = StatsAggregator(methods)
    assert agg.methods == methods


def test_mean_aggregation():
    """Test mean computation matches paper Equation 3.1."""
    df = pd.DataFrame({
        "city": ["A", "A", "B", "B"],
        "price": [100, 200, 300, 400]
    })
    
    agg = StatsAggregator([AggregationMethod.MEAN])
    result = agg.aggregate(df, ["city"], "price", min_size=1)
    
    assert len(result) == 2
    assert "price_mean" in result.columns
    
    # Check computed means
    a_mean = result[result["city"] == "A"]["price_mean"].iloc[0]
    b_mean = result[result["city"] == "B"]["price_mean"].iloc[0]
    
    assert a_mean == 150.0  # (100 + 200) / 2
    assert b_mean == 350.0  # (300 + 400) / 2


def test_std_aggregation():
    """Test standard deviation computation."""
    df = pd.DataFrame({
        "city": ["A", "A", "A"],
        "price": [100, 200, 300]
    })
    
    agg = StatsAggregator([AggregationMethod.STD])
    result = agg.aggregate(df, ["city"], "price", min_size=1)
    
    assert "price_std" in result.columns
    
    # Check std (population std with ddof=0)
    expected_std = np.std([100, 200, 300], ddof=0)
    actual_std = result["price_std"].iloc[0]
    assert np.isclose(actual_std, expected_std)


def test_min_group_size_filter():
    """Test groups below threshold are excluded."""
    df = pd.DataFrame({
        "city": ["A", "A", "B", "C", "C", "C"],
        "price": [100, 110, 200, 300, 310, 320]
    })
    
    agg = StatsAggregator([AggregationMethod.MEAN, AggregationMethod.COUNT])
    result = agg.aggregate(df, ["city"], "price", min_size=3)
    
    # Only city C has >= 3 records
    assert len(result) == 1
    assert result["city"].iloc[0] == "C"


def test_multiple_aggregations():
    """Test multiple aggregation methods work together."""
    df = pd.DataFrame({
        "city": ["A", "A", "A"],
        "price": [100, 200, 300]
    })
    
    methods = [
        AggregationMethod.MEAN,
        AggregationMethod.MEDIAN,
        AggregationMethod.MIN,
        AggregationMethod.MAX
    ]
    
    agg = StatsAggregator(methods)
    result = agg.aggregate(df, ["city"], "price", min_size=1)
    
    assert "price_mean" in result.columns
    assert "price_median" in result.columns
    assert "price_min" in result.columns
    assert "price_max" in result.columns
    
    assert result["price_mean"].iloc[0] == 200.0
    assert result["price_median"].iloc[0] == 200.0
    assert result["price_min"].iloc[0] == 100.0
    assert result["price_max"].iloc[0] == 300.0


def test_global_aggregation():
    """Test aggregation without grouping (global stats)."""
    df = pd.DataFrame({
        "price": [100, 200, 300, 400]
    })
    
    agg = StatsAggregator([AggregationMethod.MEAN])
    result = agg.aggregate(df, [], "price")
    
    assert len(result) == 1
    assert "price_mean" in result.columns
    assert result["price_mean"].iloc[0] == 250.0


def test_global_aggregation_with_all_methods():
    """Test global stats with all aggregation methods (regression test for KeyError bug)."""
    df = pd.DataFrame({"price": [100, 200, 300, 400]})
    
    methods = [
        AggregationMethod.MEAN,
        AggregationMethod.MEDIAN,
        AggregationMethod.STD,
        AggregationMethod.MIN,
        AggregationMethod.MAX,
        AggregationMethod.COUNT
    ]
    
    agg = StatsAggregator(methods)
    result = agg.aggregate(df, [], "price")
    
    assert len(result) == 1
    assert "price_mean" in result.columns
    assert "price_median" in result.columns
    assert "price_std" in result.columns
    assert "price_min" in result.columns
    assert "price_max" in result.columns
    assert "price_count" in result.columns
    
    # Verify computed values
    assert result["price_mean"].iloc[0] == 250.0
    assert result["price_median"].iloc[0] == 250.0
    assert result["price_min"].iloc[0] == 100.0
    assert result["price_max"].iloc[0] == 400.0
    assert result["price_count"].iloc[0] == 4


def test_compute_hierarchical_aggregations(sample_data):
    """Test compute_aggregations for full hierarchy."""
    stats_dict = compute_aggregations(
        df=sample_data,
        categorical_cols=["city", "neighborhood"],
        target_col="price",
        methods=[AggregationMethod.MEAN, AggregationMethod.STD],
        min_group_size=3
    )
    
    # Check expected levels (global + each categorical + interaction)
    assert "global" in stats_dict
    assert "city" in stats_dict
    assert "neighborhood" in stats_dict
    assert "city__neighborhood" in stats_dict
    
    # Check structure
    city_stats = stats_dict["city"]
    assert "city" in city_stats.columns
    assert "price_mean" in city_stats.columns


def test_relative_features_computation(sample_data):
    """Test relative feature computation (z-score and ratio)."""
    # First compute stats
    stats_dict = compute_aggregations(
        df=sample_data,
        categorical_cols=["city"],
        target_col="price",
        methods=[AggregationMethod.MEAN, AggregationMethod.STD, AggregationMethod.MEDIAN],
        min_group_size=3
    )
    
    # Join stats back (simplified for test)
    enriched = sample_data.copy()
    city_stats = stats_dict["city"]
    
    for col in ["price_mean", "price_std", "price_median"]:
        enriched[f"city_{col}"] = enriched["city"].map(
            city_stats.set_index("city")[col]
        )
    
    # Compute relative features
    result = compute_relative_features(
        df=enriched,
        stats_dict=stats_dict,
        categorical_cols=["city"],
        target_col="price"
    )
    
    # Check z-score and ratio columns exist
    assert "city_zscore" in result.columns
    assert "city_ratio" in result.columns


def test_aggregation_with_missing_columns():
    """Test aggregation handles missing columns gracefully."""
    df = pd.DataFrame({
        "city": ["A", "B"],
        "price": [100, 200]
    })
    
    agg = StatsAggregator([AggregationMethod.MEAN])
    
    # Should raise ValueError for missing group column
    with pytest.raises(ValueError, match="Missing group columns"):
        agg.aggregate(df, ["nonexistent"], "price")
    
    # Should raise ValueError for missing value column
    with pytest.raises(ValueError, match="Value column"):
        agg.aggregate(df, ["city"], "nonexistent")


def test_aggregation_with_nans():
    """Test aggregation handles NaN values correctly."""
    df = pd.DataFrame({
        "city": ["A", "A", "A", "B", "B"],
        "price": [100, np.nan, 300, 200, 400]
    })
    
    agg = StatsAggregator([AggregationMethod.MEAN])
    result = agg.aggregate(df, ["city"], "price", min_size=2)
    
    # Check that NaNs are handled
    assert len(result) == 2
    
    # Mean should exclude NaN
    a_mean = result[result["city"] == "A"]["price_mean"].iloc[0]
    assert a_mean == 200.0  # (100 + 300) / 2


def test_count_aggregation():
    """Test COUNT aggregation method."""
    df = pd.DataFrame({
        "city": ["A", "A", "A", "B", "B"],
        "price": [100, 200, 300, 400, 500]
    })
    
    agg = StatsAggregator([AggregationMethod.COUNT])
    result = agg.aggregate(df, ["city"], "price", min_size=1)
    
    assert "price_count" in result.columns
    
    a_count = result[result["city"] == "A"]["price_count"].iloc[0]
    b_count = result[result["city"] == "B"]["price_count"].iloc[0]
    
    assert a_count == 3
    assert b_count == 2


def test_quantile_stats_added():
    """Test that quantiles (Q25, Q75) are added when requested."""
    df = pd.DataFrame({
        "city": ["A"] * 10,
        "price": list(range(100, 200, 10))
    })
    
    stats_dict = compute_aggregations(
        df=df,
        categorical_cols=["city"],
        target_col="price",
        methods=[AggregationMethod.MEAN],
        include_quantiles=True
    )
    
    city_stats = stats_dict["city"]
    
    # Check quantile columns exist
    assert "price_q25" in city_stats.columns
    assert "price_q75" in city_stats.columns


def test_hierarchical_backoff_fills_nan():
    """
    Test hierarchical backoff fills NaN from global level.
    
    Per paper Section 3.4: observations in small groups should get
    statistics from global fallback when fine-grained stats are unavailable.
    
    Note: With the move away from ordered hierarchies to flat categoricals,
    backoff now goes directly to global rather than through intermediate levels.
    """
    # Create a dataframe with stats already joined
    # Simulate: city__neighborhood level has NaN, city level has value
    df = pd.DataFrame({
        "city": ["NYC", "NYC", "LA"],
        "neighborhood": ["rare_hood", "common", "common"],
        "price": [100, 200, 300],
        # Simulated joined stats - rare_hood has NaN at fine level
        "city__neighborhood_price_mean": [np.nan, 180.0, 280.0],
        "city_price_mean": [150.0, 150.0, 300.0],
        "global_price_mean": [200.0, 200.0, 200.0],
    })
    
    stats_dict = {
        "city": pd.DataFrame({"city": ["NYC", "LA"], "price_mean": [150.0, 300.0]}),
        "city__neighborhood": pd.DataFrame({
            "city": ["NYC", "LA"],
            "neighborhood": ["common", "common"],
            "price_mean": [180.0, 280.0]
        }),
        "global": pd.DataFrame({"price_mean": [200.0]})
    }
    
    result = apply_hierarchical_backoff(
        df=df,
        stats_dict=stats_dict,
        categorical_cols=["city", "neighborhood"],
        target_col="price"
    )
    
    # The NaN at city__neighborhood level should be filled from city (150.0)
    assert result.loc[0, "city__neighborhood_price_mean"] == 150.0
    # Other values should remain unchanged
    assert result.loc[1, "city__neighborhood_price_mean"] == 180.0
    assert result.loc[2, "city__neighborhood_price_mean"] == 280.0


def test_hierarchical_backoff_falls_back_to_global():
    """Test backoff falls back to global when all hierarchy levels have NaN."""
    df = pd.DataFrame({
        "city": ["RARE_CITY"],
        "price": [100],
        "city_price_mean": [np.nan],
        "global_price_mean": [500.0],
    })
    
    stats_dict = {
        "city": pd.DataFrame({"city": ["OTHER"], "price_mean": [200.0]}),
        "global": pd.DataFrame({"price_mean": [500.0]})
    }
    
    result = apply_hierarchical_backoff(
        df=df,
        stats_dict=stats_dict,
        categorical_cols=["city"],
        target_col="price"
    )
    
    # Should fall back to global
    assert result.loc[0, "city_price_mean"] == 500.0


def test_hierarchical_backoff_depth_feature():
    """Test backoff depth feature marks fallback usage."""
    df = pd.DataFrame({
        "city": ["NYC", "NYC"],
        "neighborhood": ["rare", "common"],
        "price": [100, 200],
        "city__neighborhood_price_mean": [np.nan, 180.0],
        "city_price_mean": [150.0, 150.0],
        "global_price_mean": [200.0, 200.0],
    })

    stats_dict = {
        "city": pd.DataFrame({"city": ["NYC"], "price_mean": [150.0]}),
        "city__neighborhood": pd.DataFrame({
            "city": ["NYC"],
            "neighborhood": ["common"],
            "price_mean": [180.0],
        }),
        "global": pd.DataFrame({"price_mean": [200.0]}),
    }

    result = apply_hierarchical_backoff(
        df=df,
        stats_dict=stats_dict,
        categorical_cols=["city", "neighborhood"],
        target_col="price",
        add_backoff_depth=True,
    )

    depth_col = "city__neighborhood_backoff_depth"
    assert depth_col in result.columns
    assert result.loc[0, depth_col] == 1
    assert result.loc[1, depth_col] == 0


def test_percentile_aggregations():
    """Test Q10, Q25, Q75, Q90 percentile aggregations (Paper: 'quantiles')."""
    # Create data with known percentiles
    df = pd.DataFrame({
        "city": ["A"] * 100,
        "price": list(range(1, 101))  # 1, 2, ..., 100
    })
    
    methods = [
        AggregationMethod.Q10,
        AggregationMethod.Q25,
        AggregationMethod.Q75,
        AggregationMethod.Q90,
    ]
    agg = StatsAggregator(methods)
    result = agg.aggregate(df, ["city"], "price", min_size=1)
    
    assert "price_q10" in result.columns
    assert "price_q25" in result.columns
    assert "price_q75" in result.columns
    assert "price_q90" in result.columns
    
    # Check percentile values (0-indexed, so Q25 of 1-100 is ~25.75)
    assert abs(result["price_q10"].iloc[0] - 10.9) < 1
    assert abs(result["price_q25"].iloc[0] - 25.75) < 1
    assert abs(result["price_q75"].iloc[0] - 75.25) < 1
    assert abs(result["price_q90"].iloc[0] - 90.1) < 1


def test_dispersion_aggregations():
    """Test CV, IQR, RANGE dispersion measures (Paper: 'dispersion measures')."""
    df = pd.DataFrame({
        "city": ["A"] * 100,
        "price": list(range(1, 101))  # 1, 2, ..., 100
    })
    
    methods = [
        AggregationMethod.CV,
        AggregationMethod.IQR,
        AggregationMethod.RANGE,
        AggregationMethod.VAR,
    ]
    agg = StatsAggregator(methods)
    result = agg.aggregate(df, ["city"], "price", min_size=1)
    
    assert "price_cv" in result.columns
    assert "price_iqr" in result.columns
    assert "price_range" in result.columns
    assert "price_var" in result.columns
    
    # RANGE should be 99 (100 - 1)
    assert result["price_range"].iloc[0] == 99
    
    # IQR should be ~49.5 (Q75 - Q25)
    assert abs(result["price_iqr"].iloc[0] - 49.5) < 1
    
    # CV = std / mean, both should be positive
    assert result["price_cv"].iloc[0] > 0


def test_optimized_relative_features():
    """Test optimized relative features with percentile position."""
    df = pd.DataFrame({
        "city": ["A", "A", "A"],
        "price": [100, 200, 300],
        "city_price_mean": [200.0, 200.0, 200.0],
        "city_price_std": [81.65, 81.65, 81.65],  # population std
        "city_price_median": [200.0, 200.0, 200.0],
        "city_price_q25": [150.0, 150.0, 150.0],
        "city_price_q75": [250.0, 250.0, 250.0],
    })
    
    stats_dict = {"city": pd.DataFrame({"city": ["A"], "price_mean": [200.0]})}
    
    result = compute_relative_features(
        df=df,
        stats_dict=stats_dict,
        hierarchy=["city"],
        target_col="price"
    )
    
    # Check z-score: (y - mean) / std
    assert "city_zscore" in result.columns
    
    # Check ratio: y / median  
    assert "city_ratio" in result.columns
    
    # Check percentile position: (y - Q25) / IQR
    assert "city_pct_position" in result.columns
    
    # Check deviation from mean
    assert "city_dev_from_mean" in result.columns
    
    # Verify first row: price=100, mean=200, std=81.65
    zscore_0 = (100 - 200) / 81.65
    assert abs(result["city_zscore"].iloc[0] - zscore_0) < 0.01
    
    # Verify deviation from mean
    assert result["city_dev_from_mean"].iloc[0] == -100  # 100 - 200
