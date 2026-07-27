"""
@module: equity.diagnostics.lookahead_indicator
@depends: pandas, numpy
@exports: run_lookahead_indicator, main
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §6.4.1
@data_flow: features frame + prices -> per-indicator recompute from prices[:t]
            -> assert equality with stored feature -> JSON report

Lookahead indicator diagnostic guard (S3.3). For each indicator feature column
``f`` and each row at time ``t``, recompute ``f(t)`` from ``prices[:t]`` ONLY
(the frame sliced to rows strictly before ``t`` within the same ticker group)
and assert numerical equality with the stored feature within float64 tolerance
(``abs=1e-9``).

The leak this guard catches (FOOTGUN #1, PRD §6.4.1): an SMA window that
INCLUDES ``close[t]`` (forgot ``closed='left'`` or ``.shift(1)``). The naive
violation injection in tests uses exactly this -- an SMA built with
``closed='right'`` (the pandas default) instead of the past-only form.

NaN handling (D5): NaN in early rolling windows (warmup) is "undefined at this
row", NOT a violation. The guard skips rows where either the stored value or
the re-derived value is NaN. A row where exactly one of (stored, re-derived)
is NaN and the other is finite is a violation (a partial-window bug).

Generic recompute core (D3): the guard takes a ``column_specs`` parameter --
a mapping ``{feature_col: per-ticker single-ticker spec fn}`` where each fn
takes a single-ticker prices frame (sorted ascending by ``period_close_ts``)
and returns the NAIVE (current-row-inclusive) indicator Series. The guard feeds
``prices[:t]`` through the fn and takes the last value. Defaults to
:data:`equity.features.technical.NAIVE_INDICATOR_SPECS` so sentiment-roll
features can be added later by extending the specs dict -- no guard rewrite.

CLI:

    python -m equity.diagnostics.lookahead_indicator \\
        --features <features.parquet> [--prices <prices.parquet>] \\
        [--output <report.json>]

Exits 0 on PASS, 1 on any violation. Mirrors the
:mod:`equity.diagnostics.sentiment_aggregate_guard` CLI shape (argparse,
``main()``, ``RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics" / "equity"``,
``_resolve_under_project_root`` containment guard).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from equity.data.registry import PROJECT_ROOT
from equity.features.technical import NAIVE_INDICATOR_SPECS

RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics" / "equity"

# float64 tolerance for indicator recompute equality. Indicators like SMA/EMA
# are stable to ~1e-12; MACD has cumulative drift but stays well within 1e-9
# for the window sizes we use. Use abs=1e-9 as the PRD specifies.
_ABS_TOL = 1e-9


def _resolve_under_project_root(path: str | Path) -> Path:
    """Resolve ``--output`` safely. Mirrors the sentiment_aggregate_guard
    containment guard (refuses ``..`` traversal). See
    :func:`equity.diagnostics.sentiment_aggregate_guard._resolve_under_project_root`.
    """
    raw = Path(path)
    parts = list(raw.parts)
    if ".." in parts:
        raise ValueError(f"Refusing --output with path-traversal component '..': {path}.")
    if not raw.is_absolute():
        return (PROJECT_ROOT / raw).resolve()
    return raw.resolve()


def _rederive_indicator_at_t(
    spec_fn: Callable[[pd.DataFrame], pd.Series],
    prices_ticker_pre_t: pd.DataFrame,
) -> float:
    """Recompute the NAIVE indicator on ``prices_ticker_pre_t`` (a single-ticker
    prices frame containing rows strictly before ``t``) and return the LAST
    value. The past-only feature at row ``t`` is the naive value computed at
    ``t-1`` (see :mod:`equity.features.technical`); feeding ``prices[:t]``
    through the naive fn and taking the last row reproduces that exactly.
    """
    if prices_ticker_pre_t.empty:
        return np.nan
    naive = spec_fn(prices_ticker_pre_t)
    if naive.empty:
        return np.nan
    return float(naive.iloc[-1])


def run_lookahead_indicator(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    column_specs: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
    abs_tol: float = _ABS_TOL,
) -> dict[str, Any]:
    """Run the lookahead indicator guard and return a result dict.

    Parameters
    ----------
    features:
        Feature matrix produced by :func:`equity.features.build_features` (or
        :func:`add_technical_features`). Must carry ``ticker``,
        ``period_close_ts``, and each feature column in ``column_specs``.
    prices:
        The canonical S1 prices frame used to build ``features``. Must carry
        ``ticker``, ``period_close_ts``, and the price columns each spec fn
        reads (typically ``close``, ``high``, ``low``, ``volume``).
    column_specs:
        Optional mapping ``{feature_col: per-ticker spec fn}``. Defaults to
        :data:`equity.features.technical.NAIVE_INDICATOR_SPECS`. Each fn takes
        a single-ticker prices frame (sorted ascending by ``period_close_ts``)
        and returns the NAIVE (current-row-inclusive) indicator Series.
    abs_tol:
        Absolute tolerance for float64 equality (default ``1e-9``).

    Returns
    -------
    dict
        ``{pass, n_violations, violations, n_rows, n_features_checked, ...}``.
    """
    specs = column_specs if column_specs is not None else NAIVE_INDICATOR_SPECS
    violations: list[dict[str, Any]] = []

    if "ticker" not in features.columns or "period_close_ts" not in features.columns:
        raise ValueError(
            "run_lookahead_indicator: features frame must have 'ticker' and 'period_close_ts' columns."
        )
    if "ticker" not in prices.columns or "period_close_ts" not in prices.columns:
        raise ValueError(
            "run_lookahead_indicator: prices frame must have 'ticker' and 'period_close_ts' columns."
        )

    # Pre-sort prices per ticker ascending by period_close_ts. The recompute
    # core slices ``prices[:t]`` per row, so we need O(log N) lookup by ts.
    prices_sorted = prices.sort_values(["ticker", "period_close_ts"]).reset_index(drop=True)
    # Build per-ticker (timestamp -> integer position in the sorted ticker
    # sub-frame) for O(1) "rows strictly before t" slicing.
    price_groups = {
        t: g.reset_index(drop=True) for t, g in prices_sorted.groupby("ticker", sort=False)
    }

    # We re-derive by row position within the ticker group. The "rows strictly
    # before t" slice is rows with period_close_ts < t. Find the largest
    # position whose ts < t.
    def _rows_strictly_before(ticker: str, ts: pd.Timestamp) -> pd.DataFrame:
        g = price_groups.get(ticker)
        if g is None:
            return pd.DataFrame(columns=g.columns) if g is not None else pd.DataFrame()
        # Use searchsorted on the sorted ts column.
        ts_col = g["period_close_ts"]
        # searchsorted(side='left') returns the first index >= ts.
        idx = ts_col.searchsorted(ts, side="left")
        return g.iloc[:idx]

    # Restrict to feature columns actually present in the features frame.
    feature_cols_to_check = [c for c in specs if c in features.columns]
    n_rows_checked = 0
    n_violations = 0

    for ticker, g in features.groupby("ticker", sort=False):
        # Sort the feature group by period_close_ts to align with prices.
        g_sorted = g.sort_values("period_close_ts")
        for _, row in g_sorted.iterrows():
            ts = row["period_close_ts"]
            # Canonicalize tz for the comparison -- prices may be America/New_York
            # while features may be UTC (build_features canonicalizes). Convert
            # both to UTC for the searchsorted lookup.
            ts_for_lookup = ts
            price_g = price_groups.get(ticker)
            if price_g is None or price_g.empty:
                continue
            price_ts = price_g["period_close_ts"]
            # Convert both to UTC for the lookup if tz-aware.
            if price_ts.dt.tz is not None and ts_for_lookup.tz is not None:
                if price_ts.dt.tz != ts_for_lookup.tz:
                    ts_for_lookup = ts_for_lookup.tz_convert(price_ts.dt.tz)
            pre = _rows_strictly_before(ticker, ts_for_lookup)
            n_rows_checked += 1
            for fcol in feature_cols_to_check:
                stored_val = row.get(fcol, np.nan)
                spec_fn = specs[fcol]
                rederived = _rederive_indicator_at_t(spec_fn, pre)
                # NaN handling (D5): both NaN = "undefined at this row" -> skip.
                # Exactly one NaN = partial-window bug -> violation.
                if pd.isna(stored_val) and pd.isna(rederived):
                    continue
                if pd.isna(stored_val) != pd.isna(rederived):
                    violations.append(
                        {
                            "type": "lookahead_nan_mismatch",
                            "ticker": ticker,
                            "period_close_ts": str(ts),
                            "feature": fcol,
                            "stored": None if pd.isna(stored_val) else float(stored_val),
                            "rederived": None if pd.isna(rederived) else float(rederived),
                        }
                    )
                    n_violations += 1
                    continue
                diff = abs(float(stored_val) - float(rederived))
                if diff > abs_tol:
                    violations.append(
                        {
                            "type": "lookahead_indicator_mismatch",
                            "ticker": ticker,
                            "period_close_ts": str(ts),
                            "feature": fcol,
                            "stored": float(stored_val),
                            "rederived": float(rederived),
                            "abs_diff": float(diff),
                            "tol": float(abs_tol),
                        }
                    )
                    n_violations += 1

    return {
        "pass": n_violations == 0,
        "n_violations": n_violations,
        "violations": violations,
        "n_rows_checked": n_rows_checked,
        "n_features_checked": len(feature_cols_to_check),
        "features_checked": feature_cols_to_check,
        "abs_tol": float(abs_tol),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Lookahead indicator guard (S3.3): recomputes each technical "
            "indicator from prices[:t] only and asserts equality with the "
            "stored feature. Catches SMA/EMA/RSI/etc. windows that include "
            "the current row (forgot closed='left' or .shift(1)). Exits 0 on "
            "PASS, 1 on any violation."
        ),
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Path to the feature matrix parquet (output of build_features).",
    )
    parser.add_argument(
        "--prices",
        required=True,
        help="Path to the canonical S1 prices parquet used to build the features.",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=_ABS_TOL,
        help=f"Absolute tolerance for float64 equality (default {_ABS_TOL}).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON result (resolved under PROJECT_ROOT).",
    )
    args = parser.parse_args()

    features = pd.read_parquet(args.features)
    prices = pd.read_parquet(args.prices)

    result = run_lookahead_indicator(features, prices, abs_tol=args.abs_tol)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = _resolve_under_project_root(args.output)
    else:
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"lookahead_indicator_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nResult written to: {out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
