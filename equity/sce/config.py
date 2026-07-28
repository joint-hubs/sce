"""
@module: equity.sce.config
@depends: dataclasses, typing
@exports: EquityHierarchyConfig, DEFAULT_INTERACTIONS, DEFAULT_EQUITY_HIERARCHY
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §6.1 (S4 hierarchy)
@data_flow: frozen hierarchy knobs -> EquityContextEnricher -> sce.ContextConfig

S4 equity SCE hierarchy config. Frozen knobs that the equity wrapper maps
onto upstream ``sce.ContextConfig``. The interaction allow-list is *equity*
policy (see ADR 0001); upstream SCE still expands all pairs and the enricher
post-filters to this list (interim until an upstream ``interactions`` field
lands).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Curated interaction pairs for the equity manifold. ``ticker`` is a leaf
# group key (single-col level only) — ticker×anything explodes cardinality and
# is redundant with the ticker leaf, so it is intentionally absent here.
DEFAULT_INTERACTIONS: Tuple[Tuple[str, ...], ...] = (
    ("sector", "time_bucket"),
    ("industry", "time_bucket"),
    ("mktcap_bucket", "time_bucket"),
    ("sector", "mktcap_bucket"),
)


@dataclass(frozen=True)
class EquityHierarchyConfig:
    """Frozen knobs for equity SCE enrichment (FOC-51 S4.1).

    Defaults match the locked S4 decisions:
    ``target_col="ret_1d"``, rolling cross-fit, ``min_group_size=20``,
    ``include_relative_features=False`` (relative features leak the target),
    curated ``interactions`` allow-list (ADR 0001).
    """

    target_col: str = "ret_1d"
    categorical_cols: Tuple[str, ...] = (
        "ticker",
        "sector",
        "industry",
        "mktcap_bucket",
        "time_bucket",
    )
    interactions: Tuple[Tuple[str, ...], ...] = DEFAULT_INTERACTIONS
    min_group_size: int = 20
    cross_fit_strategy: str = "rolling"
    time_col: str = "period_close_ts"
    n_folds: int = 5
    random_state: int = 42
    max_interaction_depth: int = 2
    include_relative_features: bool = False
    # Defensive: drop any forward-looking ``ret_h*`` targets if present on the
    # features frame so auto-detection / SCE never treats them as inputs.
    exclude_cols: Tuple[str, ...] = ()


DEFAULT_EQUITY_HIERARCHY: EquityHierarchyConfig = EquityHierarchyConfig()
