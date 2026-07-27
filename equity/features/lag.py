"""
@module: equity.features.lag
@depends: pandas, numpy
@exports: apply_lags, LagConfig, DEFAULT_LAG_WINDOWS
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §5.3 (S3.2)
@data_flow: feature columns -> per-(ticker) lag/rollmean/rollstd transforms

S3.2 lag layer. Feature-source-agnostic (D1): works on ANY base column,
technical OR sentiment. For each base column ``c`` and each lag window ``N`` in
``L = (1, 3, 5, 10, 21)`` emits:

* ``{c}_lag{N}``       -- ``c.shift(N)`` (value N rows back, within ticker).
* ``{c}_rollmean{N}``  -- rolling mean over N rows, past-only.
* ``{c}_rollstd{N}``   -- rolling std over N rows, past-only.

Past-only (FOOTGUN #1 / D6): ``shift(N)`` is inherently past-only (value N rows
back). The rolling mean/std use ``closed='left'`` EXPLICITLY (excludes the
current row) so the value at row ``t`` reflects rows ``[t-N, t-1]`` only -- no
current-row leakage. The naive pandas ``rolling(window).mean()`` defaults to
``closed='right'`` (INCLUDES current row) and is the exact leak the
:mod:`equity.diagnostics.lookahead_indicator` guard catches.

Per-ticker independence (D7): every rolling op is applied via
``groupby("ticker", sort=False)`` so a window never bleeds across the ticker
boundary. Inputs must be sorted ascending by ``period_close_ts`` within
ticker; ``apply_lags`` sorts defensively.

NaN in early windows (D5): pandas emits native NaN during warmup; we do NOT
mask. ``_lag{N}`` is NaN for the first N rows of each ticker; ``_rollmean{N}``
and ``_rollstd{N}`` are NaN for the first N rows (rolling needs N past rows).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

DEFAULT_LAG_WINDOWS: tuple[int, ...] = (1, 3, 5, 10, 21)


@dataclass(frozen=True)
class LagConfig:
    """Configuration for the lag layer.

    ``windows`` = lag windows ``L`` (PRD §5.3 default ``(1, 3, 5, 10, 21)``).
    ``methods`` controls which transforms are emitted per (base_col, window).
    """

    windows: tuple[int, ...] = DEFAULT_LAG_WINDOWS
    methods: tuple[str, ...] = ("shift", "rolling_mean", "rolling_std")

    def describe(self, base_columns: Iterable[str]) -> list[str]:
        """Return the list of output column names emitted for ``base_columns``."""
        cols: list[str] = []
        for c in base_columns:
            for w in self.windows:
                if "shift" in self.methods:
                    cols.append(f"{c}_lag{w}")
                if "rolling_mean" in self.methods:
                    cols.append(f"{c}_rollmean{w}")
                if "rolling_std" in self.methods:
                    cols.append(f"{c}_rollstd{w}")
        return cols


def _shift_lag(s: pd.Series, n: int) -> pd.Series:
    """``s.shift(n)`` -- value n rows back. Inherently past-only for n >= 1."""
    return s.shift(n)


def _rollmean_past(s: pd.Series, n: int) -> pd.Series:
    """Rolling mean over n rows, EXCLUDING current row (``closed='left'``).

    At row ``t``: mean of ``s[t-n .. t-1]`` (N past rows). The explicit
    ``closed='left'`` is the past-only guard (FOOTGUN #1) -- it would catch a
    naive ``rolling(n).mean()`` that defaults to ``closed='right'`` and
    includes ``s[t]``.
    """
    return s.rolling(n, closed="left").mean()


def _rollstd_past(s: pd.Series, n: int) -> pd.Series:
    """Rolling std over n rows, EXCLUDING current row (``closed='left'``)."""
    return s.rolling(n, closed="left").std()


def apply_lags(
    df: pd.DataFrame,
    columns: Iterable[str],
    *,
    windows: Iterable[int] = DEFAULT_LAG_WINDOWS,
    methods: Iterable[str] = ("shift", "rolling_mean", "rolling_std"),
) -> pd.DataFrame:
    """Apply lag/rollmean/rollstd transforms to ``columns`` per ticker.

    For each base column ``c`` and window ``N`` in ``windows``, emits:

    * ``{c}_lag{N}``      iff ``"shift"`` in ``methods``
    * ``{c}_rollmean{N}`` iff ``"rolling_mean"`` in ``methods``
    * ``{c}_rollstd{N}``  iff ``"rolling_std"`` in ``methods``

    All transforms are per-ticker (D7) and past-only (D6). The input ``df`` MUST
    carry a ``ticker`` column; if it lacks one, a ValueError is raised.

    Parameters
    ----------
    df:
        Feature frame with ``ticker`` and ``period_close_ts`` columns plus the
        base ``columns``. Sorted ascending by ``period_close_ts`` within ticker
        (this function sorts defensively).
    columns:
        Base columns to lag. Must exist in ``df``.
    windows:
        Lag windows ``L`` (default ``(1, 3, 5, 10, 21)``).
    methods:
        Which transforms to emit per (col, window).

    Returns
    -------
    pd.DataFrame
        ``df`` with lag columns appended. Does NOT mutate the input.
    """
    if "ticker" not in df.columns:
        raise ValueError("apply_lags requires a 'ticker' column for per-ticker grouping (D7).")
    base_cols = list(columns)
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        raise ValueError(f"apply_lags: base columns not found in df: {missing}")
    win_list = list(windows)
    meth = set(methods)

    out = df.copy()
    if out.empty:
        return out
    # Defensive per-ticker sort (D7 prelude): every rolling/shift op below
    # relies on ascending time order within each ticker group.
    if "period_close_ts" in out.columns:
        out = out.sort_values(["ticker", "period_close_ts"]).reset_index(drop=True)

    for c in base_cols:
        grouped = out.groupby("ticker", sort=False)[c]
        for w in win_list:
            if "shift" in meth:
                out[f"{c}_lag{w}"] = grouped.transform(lambda s, n=w: _shift_lag(s, n))
            if "rolling_mean" in meth:
                out[f"{c}_rollmean{w}"] = grouped.transform(lambda s, n=w: _rollmean_past(s, n))
            if "rolling_std" in meth:
                out[f"{c}_rollstd{w}"] = grouped.transform(lambda s, n=w: _rollstd_past(s, n))
    return out


__all__ = ["apply_lags", "LagConfig", "DEFAULT_LAG_WINDOWS"]
