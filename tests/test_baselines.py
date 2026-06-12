"""Tests for context baseline variant resolution."""

import pytest

from sce.baselines import (
    get_context_variant_label,
    normalize_context_variant,
    resolve_context_variant_methods,
)
from sce.config import AggregationMethod


def test_target_mean_variant_resolves_to_mean_only():
    methods = resolve_context_variant_methods(
        "target_mean",
        [AggregationMethod.MEAN, AggregationMethod.STD, AggregationMethod.COUNT],
    )

    assert methods == [AggregationMethod.MEAN]


def test_mean_std_count_variant_resolves_expected_statistics():
    methods = resolve_context_variant_methods(
        "hierarchical_mean_std_count", [AggregationMethod.MEAN]
    )

    assert methods == [AggregationMethod.MEAN, AggregationMethod.STD, AggregationMethod.COUNT]


def test_sce_variant_preserves_default_methods():
    defaults = [AggregationMethod.MEAN, AggregationMethod.MEDIAN]
    methods = resolve_context_variant_methods("sce", defaults)

    assert methods == defaults


def test_context_variant_validation_and_labels():
    assert normalize_context_variant("TARGET_MEAN") == "target_mean"
    assert get_context_variant_label("hierarchical_mean_count") == "Hierarchical Mean+Count"

    with pytest.raises(ValueError, match="Unsupported context variant"):
        normalize_context_variant("unknown")
