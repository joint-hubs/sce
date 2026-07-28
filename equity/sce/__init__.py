"""
@module: equity.sce
@depends: pandas, sce
@exports: EquityContextEnricher, EquityHierarchyConfig, build_context_config,
          DEFAULT_EQUITY_HIERARCHY, DEFAULT_INTERACTIONS
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §6.1 (S4)
@data_flow: features frame + static sectors -> EquityContextEnricher
            -> rolling cross-fitted SCE context columns (allow-list filtered)

S4 equity SCE enrichment package. Re-exports are LAZY so
``import equity.sce`` stays light (pandas / sce loaded only when a symbol is
actually accessed). Mirrors ``equity.diagnostics`` / ``equity.features``.
"""

from __future__ import annotations


def __getattr__(name: str):  # pragma: no cover - thin re-export
    if name in {
        "EquityHierarchyConfig",
        "DEFAULT_EQUITY_HIERARCHY",
        "DEFAULT_INTERACTIONS",
    }:
        from equity.sce import config as _config

        return getattr(_config, name)
    if name in {
        "EquityContextEnricher",
        "build_context_config",
    }:
        from equity.sce import enrich as _enrich

        return getattr(_enrich, name)
    raise AttributeError(f"module 'equity.sce' has no attribute {name!r}")


__all__ = [
    "EquityContextEnricher",
    "EquityHierarchyConfig",
    "DEFAULT_EQUITY_HIERARCHY",
    "DEFAULT_INTERACTIONS",
    "build_context_config",
]
