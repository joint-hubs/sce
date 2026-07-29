"""
@module: equity.forecaster.targets
@depends: numpy, pandas
@exports: add_forward_targets, forward_target_col, DEFAULT_HORIZONS
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §7.1, §9.3
@data_flow: prices(close,ticker,period_close_ts) -> ret_hN label columns (LABEL ONLY)

Forward N-period log-return targets for the multi-horizon forecaster.

``ret_hN[t] = log(close[t+N] / close[t])`` per ticker. These are LABELS —
they must NEVER enter a model design matrix or SCE regime (see
:mod:`equity.diagnostics.forward_target_isolation` and the defensive drop in
:meth:`equity.sce.enrich.EquityContextEnricher._prepare`).

This family is distinct from past-only features ``ret_{N}d_log`` emitted by
:func:`equity.features.technical.add_returns`.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd

from equity.forecaster.config import DEFAULT_HORIZONS


def forward_target_col(horizon: int) -> str:
    """Return the canonical forward-target column name for ``horizon``."""
    if int(horizon) < 1:
        raise ValueError(f"horizon must be >= 1; got {horizon}")
    return f"ret_h{int(horizon)}"


def add_forward_targets(
    prices: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    close_col: str = "close",
    ticker_col: str = "ticker",
    time_col: str = "period_close_ts",
) -> pd.DataFrame:
    """Append forward log-return targets ``ret_hN`` per ticker.

    Parameters
    ----------
    prices:
        Panel with at least ``ticker``, ``period_close_ts``, ``close``. Does not
        need to be pre-sorted; the function sorts defensively per ticker.
    horizons:
        Positive integers N for which to emit ``ret_hN = log(close[t+N]/close[t])``.
    close_col, ticker_col, time_col:
        Column names.

    Returns
    -------
    pd.DataFrame
        Copy of ``prices`` (re-ordered by ticker / time) with one new column per
        horizon. Tail rows within each ticker (last N observations) are NaN
        because the future close is unknown — callers must dropna on the active
        horizon before training.
    """
    if close_col not in prices.columns:
        raise ValueError(f"add_forward_targets: missing close column {close_col!r}")
    if ticker_col not in prices.columns:
        raise ValueError(f"add_forward_targets: missing ticker column {ticker_col!r}")
    if time_col not in prices.columns:
        raise ValueError(f"add_forward_targets: missing time column {time_col!r}")

    hs: Tuple[int, ...] = tuple(int(h) for h in horizons)
    if not hs:
        raise ValueError("add_forward_targets: horizons must be non-empty")
    if any(h < 1 for h in hs):
        raise ValueError(f"add_forward_targets: every horizon must be >= 1; got {hs}")

    out = prices.sort_values([ticker_col, time_col]).reset_index(drop=True).copy()
    for h in hs:
        col = forward_target_col(h)
        # shift(-h) pulls the future close down onto row t; log-ratio is causal
        # as a LABEL of the future, never as a feature of the present.
        future = out.groupby(ticker_col, sort=False)[close_col].shift(-h)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[col] = np.log(future.astype(float) / out[close_col].astype(float))
    return out


def list_forward_target_cols(horizons: Iterable[int] = DEFAULT_HORIZONS) -> Tuple[str, ...]:
    """Return ``(ret_h1, ret_h5, ...)`` for the given horizons."""
    return tuple(forward_target_col(h) for h in horizons)
