"""
@module: sce.baselines
@depends: sce.config
@exports: SUPPORTED_CONTEXT_VARIANTS, get_context_variant_label, normalize_context_variant, resolve_context_variant_methods
@data_flow: context variant name -> aggregation subset
"""

from __future__ import annotations

from typing import Iterable

from sce.config import AggregationMethod

_CONTEXT_VARIANT_METHODS: dict[str, tuple[AggregationMethod, ...] | None] = {
    "sce": None,
    "target_mean": (AggregationMethod.MEAN,),
    "hierarchical_mean_count": (AggregationMethod.MEAN, AggregationMethod.COUNT),
    "hierarchical_mean_std_count": (
        AggregationMethod.MEAN,
        AggregationMethod.STD,
        AggregationMethod.COUNT,
    ),
}

_CONTEXT_VARIANT_LABELS = {
    "sce": "SCE",
    "target_mean": "Target Mean",
    "hierarchical_mean_count": "Hierarchical Mean+Count",
    "hierarchical_mean_std_count": "Hierarchical Mean+Std+Count",
}

SUPPORTED_CONTEXT_VARIANTS: tuple[str, ...] = tuple(_CONTEXT_VARIANT_METHODS)


def normalize_context_variant(context_variant: str | None) -> str:
    """Return a supported context variant name."""
    normalized = (context_variant or "sce").strip().lower()
    if normalized not in _CONTEXT_VARIANT_METHODS:
        supported = ", ".join(SUPPORTED_CONTEXT_VARIANTS)
        raise ValueError(
            f"Unsupported context variant '{context_variant}'. Supported variants: {supported}"
        )
    return normalized


def resolve_context_variant_methods(
    context_variant: str | None,
    default_methods: Iterable[AggregationMethod],
) -> list[AggregationMethod]:
    """Resolve the aggregation methods for a context variant."""
    normalized = normalize_context_variant(context_variant)
    variant_methods = _CONTEXT_VARIANT_METHODS[normalized]
    if variant_methods is None:
        return list(default_methods)
    return list(variant_methods)


def get_context_variant_label(context_variant: str | None) -> str:
    """Return a human-readable label for a context variant."""
    normalized = normalize_context_variant(context_variant)
    return _CONTEXT_VARIANT_LABELS[normalized]
