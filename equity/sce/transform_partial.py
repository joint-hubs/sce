"""
@module: equity.sce.transform_partial
@depends: pandas, equity.sce.enrich
@exports: transform_partial
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §6.1 (S4.4)
@data_flow: EquityContextEnricher + new_rows + refit_boundary_ts
            -> enricher.transform_partial (PIT-safe refit+transform)

Equity-local transform_partial wrapper (S4.4) — NOT upstream sce/.
Thin discoverability shim; the real implementation lives on
:class:`equity.sce.enrich.EquityContextEnricher`.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from .enrich import EquityContextEnricher


def transform_partial(
    enricher: EquityContextEnricher,
    new_rows: pd.DataFrame,
    *,
    refit_boundary_ts: Any,
    train_start: Optional[Any] = None,
) -> pd.DataFrame:
    """Re-fit the SCE engine on ``[train_start, refit_boundary_ts]`` then transform new_rows.

    PIT-safe: new_rows context stats come ONLY from the fit window.
    See :meth:`EquityContextEnricher.transform_partial` for full semantics.
    """
    return enricher.transform_partial(
        new_rows,
        refit_boundary_ts=refit_boundary_ts,
        train_start=train_start,
    )


__all__ = ["transform_partial"]
