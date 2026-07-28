"""
@module: tests.equity.test_sce_hierarchy
@depends: equity.sce.config, equity.sce.enrich, sce.ContextConfig
@exports:
@data_flow: EquityHierarchyConfig -> build_context_config / _allowed_levels

S4.1 hierarchy config unit tests (frozen knobs + ContextConfig mapping +
interaction allow-list levels).
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from equity.sce import (
    DEFAULT_INTERACTIONS,
    EquityContextEnricher,
    EquityHierarchyConfig,
    build_context_config,
)
from sce import ContextConfig


def test_hierarchy_config_is_frozen() -> None:
    cfg = EquityHierarchyConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.min_group_size = 99  # type: ignore[misc]


def test_default_interactions_curated() -> None:
    # ticker must NOT appear in any interaction pair (leaf key only).
    for combo in DEFAULT_INTERACTIONS:
        assert "ticker" not in combo
    assert ("sector", "time_bucket") in DEFAULT_INTERACTIONS
    assert ("industry", "time_bucket") in DEFAULT_INTERACTIONS
    assert ("mktcap_bucket", "time_bucket") in DEFAULT_INTERACTIONS
    assert ("sector", "mktcap_bucket") in DEFAULT_INTERACTIONS


def test_build_context_config_locked_fields() -> None:
    cfg = build_context_config(EquityHierarchyConfig())
    assert isinstance(cfg, ContextConfig)
    assert cfg.target_col == "ret_1d"
    assert cfg.min_group_size == 20
    assert cfg.cross_fit_strategy == "rolling"
    assert cfg.time_col == "period_close_ts"
    assert cfg.include_relative_features is False
    assert cfg.use_cross_fitting is True
    assert cfg.n_folds == 5
    assert cfg.random_state == 42
    assert cfg.max_interaction_depth == 2
    assert cfg.include_interactions is True
    assert cfg.include_global_stats is True
    assert cfg.categorical_cols == [
        "ticker",
        "sector",
        "industry",
        "mktcap_bucket",
        "time_bucket",
    ]


def test_allowed_levels_contains_curated_not_ticker_pairs() -> None:
    enricher = EquityContextEnricher(
        hierarchy=EquityHierarchyConfig(),
        # sectors unused for _allowed_levels; pass a minimal valid frame so
        # __init__ does not try to read the on-disk CSV.
        sectors=_minimal_sectors(),
    )
    levels = enricher._allowed_levels()
    assert "global" in levels
    assert "ticker" in levels
    assert "sector" in levels
    assert "industry" in levels
    assert "mktcap_bucket" in levels
    assert "time_bucket" in levels
    assert "sector__time_bucket" in levels
    assert "industry__time_bucket" in levels
    assert "mktcap_bucket__time_bucket" in levels
    assert "sector__mktcap_bucket" in levels
    # Disallowed noise pairs must NOT appear.
    assert "ticker__sector" not in levels
    assert "ticker__industry" not in levels
    assert "ticker__time_bucket" not in levels
    assert "ticker__mktcap_bucket" not in levels


def _minimal_sectors():
    import pandas as pd

    return pd.DataFrame(
        {
            "ticker": ["TK0"],
            "sector": ["Information Technology"],
            "industry": ["Systems Software"],
            "mktcap_bucket": ["large"],
        }
    )
