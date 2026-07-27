"""``published_at_guard`` -- point-in-time leakage diagnostic for S1.3.

This module is a **join-invariant sanity check**, not a feed-side leakage
detector. The guard asserts three invariants on the joined frame produced by
:meth:`equity.data.loader.EquityDataLoader.join_articles_to_prices`:

1. **Right-hand side** -- ``published_at <= period_close_ts`` for every row
   (an article is not assigned to a period that closed before it was
   published). NOTE: because the join uses ``get_indexer(method="bfill")``,
   this inequality holds by construction for every row the join produces;
   the check exists to catch regressions in the join implementation (e.g. a
   switch to ``method='ffill'``) and downstream tampering, NOT to detect
   feed-side backdating (a feed that backdates ``published_at`` cannot be
   caught here -- the value is already a canonical UTC timestamp by the time
   the join runs).

2. **Left-hand side** -- ``period_close(P-1) < published_at`` (the previous
   stored session close is strictly before ``published_at``, OR the row is
   the first stored session). This catches an ``ffill`` regression where an
   article published between two sessions is wrongly bound to the earlier
   session.

3. **Foreign-key integrity** (optional, when a prices frame is provided):
   every ``period_close_ts`` in the joined frame must exist in the prices
   store's ``period_close_ts`` set. A violation means the join produced a
   phantom key -- raised as :class:`ValueError` (CLI exits 1).

Both timestamps are compared in UTC (the price-side ``period_close_ts`` is
stored tz-aware ``America/New_York`` / canonicalized to UTC in the joined
output, and is coerced to UTC here; ``published_at`` is already UTC). The
comparison is therefore DST-safe (no wall-clock arithmetic across DST
transitions).

Run as a module:

    python -m equity.diagnostics.published_at_guard --prices <dir> --articles <dir>

or, when a pre-joined parquet already exists:

    python -m equity.diagnostics.published_at_guard --joined <file>

The CLI exits with code 0 on PASS and 1 on any violation (DoD: non-zero on a
synthetic injected violation). This DIVERGES from the rest of SCE's
``scripts/diagnostics/`` family, which rely on uncaught exceptions for failure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equity.data.registry import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics" / "equity"


def _coerce_utc(series: pd.Series, col: str) -> pd.Series:
    """Coerce a tz-aware timestamp Series to UTC; raise on tz-naive."""
    if series.dt.tz is None:
        raise ValueError(
            f"{col} is tz-naive in the joined frame; expected tz-aware "
            "(America/New_York or UTC). Refusing to compare (potential "
            "point-in-time leak)."
        )
    return series.dt.tz_convert("UTC")


def assert_fk_integrity(joined_df: pd.DataFrame, prices_df: pd.DataFrame) -> None:
    """Assert every ``period_close_ts`` in ``joined_df`` exists in the prices
    store's ``period_close_ts`` set.

    This catches phantom foreign keys -- sessions the join produced that have
    no corresponding price row. Both columns are coerced to UTC before the
    membership check. Raises :class:`ValueError` on the first violation with a
    small sample of the offending keys.
    """
    if "period_close_ts" not in prices_df.columns:
        raise ValueError(
            "prices_df missing required column 'period_close_ts' for FK check."
        )
    if joined_df.empty:
        return
    pc = _coerce_utc(pd.Series(joined_df["period_close_ts"]), "period_close_ts")
    prices_ts = pd.Series(prices_df["period_close_ts"])
    if prices_ts.dt.tz is None:
        raise ValueError(
            "prices_df.period_close_ts is tz-naive; expected tz-aware "
            "America/New_York."
        )
    prices_ts_utc = prices_ts.dt.tz_convert("UTC")
    prices_keys = set(prices_ts_utc.unique())
    bad_mask = ~pc.isin(prices_keys)
    if bad_mask.any():
        bad = pc[bad_mask].head(5).tolist()
        raise ValueError(
            f"FK integrity violated: {int(bad_mask.sum())} joined "
            f"period_close_ts value(s) not present in the prices store. "
            f"First 5 (UTC): {[str(ts) for ts in bad]}"
        )


def run_published_at_guard(
    joined_df: pd.DataFrame,
    prices_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Assert the PIT invariants on ``joined_df`` (and optionally FK
    integrity against ``prices_df``).

    Both timestamps are compared in UTC. ``period_close_ts`` may be tz-aware
    ``America/New_York`` (the canonical price-side dtype) or already UTC; it
    is coerced to UTC. ``published_at`` is expected to be tz-aware UTC (the
    canonical articles dtype); a tz-naive column raises ``ValueError`` (it
    indicates a schema slip upstream).

    Parameters
    ----------
    joined_df:
        DataFrame with at least the columns ``ticker``, ``period_close_ts``,
        ``published_at``, ``text``, ``source`` -- the output shape of
        :meth:`equity.data.loader.EquityDataLoader.join_articles_to_prices`.
    prices_df:
        Optional prices DataFrame. When provided, an FK integrity check is
        run (see :func:`assert_fk_integrity`); a violation raises
        :class:`ValueError` and propagates to the caller (the CLI converts it
        to exit code 1).

    Returns
    -------
    dict
        ``{"pass": bool, "n_violations": int, "violations": list[dict],
        "n_checked": int, "fk_checked": bool}``. Each violation dict carries
        the row's ``ticker``, ``published_at`` (ISO UTC), ``period_close_ts``
        (ISO UTC) and the gap in seconds (positive => leak).
    """
    required = {"ticker", "period_close_ts", "published_at", "text", "source"}
    missing = required - set(joined_df.columns)
    if missing:
        raise ValueError(f"joined_df missing required columns: {sorted(missing)}")

    # FK integrity check (raises ValueError on violation when prices provided).
    fk_checked = False
    if prices_df is not None:
        assert_fk_integrity(joined_df, prices_df)
        fk_checked = True

    if joined_df.empty:
        return {
            "pass": True,
            "n_violations": 0,
            "violations": [],
            "n_checked": 0,
            "fk_checked": fk_checked,
        }

    pc = pd.Series(joined_df["period_close_ts"]).copy()
    pc_utc = _coerce_utc(pc, "period_close_ts")

    pub = pd.Series(joined_df["published_at"]).copy()
    pub_utc = _coerce_utc(pub, "published_at")

    # ---- Right-hand side: published_at <= period_close_ts ----------------
    # Holds by construction for bfill-derived joins; catches ffill regressions
    # and downstream tampering.
    gap = pub_utc.to_numpy() - pc_utc.to_numpy()  # timedelta64[ns]
    rhs_violation = gap > pd.Timedelta(0)

    # ---- Left-hand side: period_close(P-1) < published_at -----------------
    # The previous stored session close must be strictly before published_at
    # (OR the row is the first stored session -- no previous session exists).
    # Vectorized (review round 2, issue #3): the per-row Python loop is
    # replaced with a single ``np.where`` over numpy arrays -- the same LHS
    # semantics (``sorted_closes[close_pos - 1] >= pub_utc`` when there IS a
    # previous session, ``False`` otherwise). Preserves the R1 m1 regression
    # contract exactly.
    sorted_closes = pd.DatetimeIndex(sorted(set(pc_utc)))
    close_pos = sorted_closes.get_indexer(pc_utc.to_numpy(), method="pad")
    # Both sides are tz-aware UTC; strip the tz to tz-naive datetime64[ns] in
    # UTC for a single vectorized numpy comparison (avoids the object-array
    # comparison failure between ``DatetimeIndex.values`` (datetime64) and
    # ``tz-aware Series.to_numpy()`` (object of Timestamps)).
    sorted_naive = sorted_closes.tz_localize(None).to_numpy()
    pub_naive = pub_utc.dt.tz_localize(None).to_numpy()
    # close_pos == 0 (first session) -> no previous -> LHS holds (not a
    # violation). close_pos == -1 (not found) -> also not a violation here
    # (the RHS check or FK check catches it; the LHS guard is scoped to the
    # "previous session" rule only). Use np.where to avoid negative-index
    # garbage where close_pos <= 0 (False-masked by np.where).
    lhs_violation = np.where(
        close_pos > 0,
        sorted_naive[np.where(close_pos > 0, close_pos - 1, 0)] >= pub_naive,
        False,
    )

    violation_mask = rhs_violation | lhs_violation

    violations: list[dict] = []
    if violation_mask.any():
        bad_idx = joined_df.index[violation_mask]
        for i in bad_idx:
            pc_ts = pd.Timestamp(pc_utc.loc[i])
            pub_ts = pd.Timestamp(pub_utc.loc[i])
            violations.append(
                {
                    "ticker": str(joined_df.loc[i, "ticker"]),
                    "published_at": pub_ts.isoformat(),
                    "period_close_ts": pc_ts.isoformat(),
                    "gap_seconds": int(
                        (pub_ts - pc_ts).total_seconds()
                    ),
                }
            )

    return {
        "pass": len(violations) == 0,
        "n_violations": len(violations),
        "violations": violations,
        "n_checked": int(len(joined_df)),
        "fk_checked": fk_checked,
    }


def _resolve_prices_path_from_universe(universe_name: str) -> Path | None:
    """Resolve the prices store path from the universe TOML's
    ``[prices].output_dir`` (review round 2, suggestion #6).

    Returns the resolved absolute path if the TOML exists and has a
    ``[prices].output_dir``; returns ``None`` if the TOML/prices section is
    missing, the universe is unknown, or the resolved path does not exist.
    The caller is expected to warn and skip FK integrity when ``None`` is
    returned (do NOT hard-fail -- the PIT check still runs).
    """
    try:
        from equity.data.registry import PROJECT_ROOT, get_universe_info

        info = get_universe_info(universe_name)
    except (FileNotFoundError, ValueError):
        return None
    try:
        import tomllib

        with info.path.open("rb") as handle:
            payload = tomllib.load(handle)
    except (OSError, ValueError):
        return None
    prices_rel = payload.get("prices", {}).get("output_dir")
    if not prices_rel:
        return None
    prices_path = Path(prices_rel)
    if not prices_path.is_absolute():
        prices_path = PROJECT_ROOT / prices_path
    if not prices_path.exists():
        return None
    return prices_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Point-in-time leakage guard: asserts published_at <= "
            "period_close_ts for every joined (article, period) row. Exits 0 "
            "on PASS, 1 on any violation."
        ),
    )
    parser.add_argument(
        "--joined",
        help=(
            "Path to a pre-joined articles_joined.parquet file. When "
            "provided, --prices/--articles are ignored (unless --prices is "
            "also given for the FK integrity check)."
        ),
    )
    parser.add_argument(
        "--prices",
        help=(
            "Path to the partitioned prices parquet store. Required when "
            "--joined is absent (the guard builds the join itself). When "
            "--joined IS provided, --prices enables the FK integrity check "
            "(every joined period_close_ts must exist in the prices store). "
            "If omitted with --joined, the prices store is resolved from the "
            "universe TOML [prices].output_dir (review round 2, suggestion #6) "
            "so the documented one-arg form also runs FK integrity; falls back "
            "to PIT-only if the TOML/prices can't be resolved."
        ),
    )
    parser.add_argument(
        "--articles",
        help="Path to the partitioned articles parquet store (used when --joined is absent).",
    )
    parser.add_argument(
        "--universe",
        default="sp500",
        help=(
            "Universe name for resolving --prices from the TOML when --joined "
            "is given without --prices (resolves configs/equity/<name>.toml). "
            "Default: 'sp500'."
        ),
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional path to write the JSON result. When omitted, the result "
            f"is written under {RESULTS_DIR} with a UTC timestamp suffix."
        ),
    )
    args = parser.parse_args()

    prices_df: pd.DataFrame | None = None
    if args.joined:
        joined = pd.read_parquet(args.joined)
        # Review round 2, suggestion #6: resolve --prices from the universe
        # TOML [prices].output_dir BY DEFAULT when --joined is given without
        # --prices, so the documented one-arg form also runs FK integrity. If
        # the TOML/prices can't be resolved, warn and run PIT-only (don't
        # hard-fail -- the PIT check is the primary invariant).
        prices_path = args.prices
        if prices_path is None:
            prices_path_resolved = _resolve_prices_path_from_universe(args.universe)
            if prices_path_resolved is not None:
                prices_path = str(prices_path_resolved)
            else:
                import logging

                logging.getLogger("equity.diagnostics.published_at_guard").warning(
                    "FK integrity check skipped: no --prices given and could "
                    "not resolve [prices].output_dir from universe '%s' TOML "
                    "(missing section, unknown universe, or path does not "
                    "exist). Run with --prices <dir> to enable FK integrity.",
                    args.universe,
                )
        if prices_path is not None:
            try:
                prices_df = pd.read_parquet(prices_path)
            except (FileNotFoundError, OSError, ValueError) as exc:
                import logging

                logging.getLogger("equity.diagnostics.published_at_guard").warning(
                    "Could not read prices store at %s for FK check (%s); "
                    "running PIT-only.",
                    prices_path,
                    exc,
                )
                prices_df = None
    else:
        if not args.prices or not args.articles:
            parser.error(
                "Either --joined OR both --prices and --articles are required."
            )
        # The join needs an EquityDataLoader with a universe + window. We
        # default to the user-provided universe (default 'sp500') and infer
        # the loader's window from the prices frame's [min, max]
        # period_close_ts. This mirrors how a user would invoke the guard on
        # a freshly-built store.
        from equity.data.loader import EquityDataLoader

        prices_df = pd.read_parquet(args.prices)
        start = pd.Timestamp(prices_df["period_close_ts"].min()).normalize()
        end = pd.Timestamp(prices_df["period_close_ts"].max()).normalize() + pd.Timedelta(days=1)
        loader = EquityDataLoader(
            args.universe,
            start.strftime("%Y-%m-%d"),
            end.strftime("%Y-%m-%d"),
        )
        joined_path = loader.join_articles_to_prices(
            prices_path=args.prices,
            articles_path=args.articles,
        )
        joined = pd.read_parquet(joined_path)

    try:
        result = run_published_at_guard(joined, prices_df=prices_df)
    except ValueError as exc:
        # FK integrity violation surfaces here as exit 1. Review round 2,
        # issue #2: populate ``violations`` with a synthetic entry so the
        # result schema is consistent across both the normal path and the
        # except path (downstream JSON consumers see one shape, not two).
        synthetic = {
            "type": "fk_error",
            "message": str(exc),
        }
        result = {
            "pass": False,
            "n_violations": 1,
            "violations": [synthetic],
            "n_checked": int(len(joined)),
            "fk_checked": prices_df is not None,
            "error": str(exc),
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = (
            Path(args.output)
            if args.output
            else RESULTS_DIR
            / f"published_at_guard_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        print(f"\nResult written to: {out_path}")
        return 1

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"published_at_guard_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nResult written to: {out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
