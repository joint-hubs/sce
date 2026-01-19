"""
Tests for configuration and validation.
"""

import pytest

from sce.config import AggregationMethod, ContextConfig


def test_config_validation():
    """Test configuration validation rules."""
    # Valid config should work
    config = ContextConfig(categorical_cols=["country", "city"], target_col="price")
    assert config.categorical_cols == ["country", "city"]

    # Empty categorical_cols is allowed (auto-detect mode)
    config_auto = ContextConfig(target_col="price")
    assert config_auto.categorical_cols is None

    # Missing target should fail
    with pytest.raises(ValueError, match="target_col must be"):
        ContextConfig(categorical_cols=["city"], target_col="")


def test_aggregation_methods():
    """Test AggregationMethod enum."""
    assert AggregationMethod.MEAN.value == "mean"
    assert AggregationMethod.STD.value == "std"
