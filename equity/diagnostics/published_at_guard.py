"""``published_at_guard`` -- point-in-time leakage diagnostic for S1.3.

The guard asserts that for every joined (article, period) row the article's
``published_at`` does not exceed the period's session close:

    published_at <= period_close_ts

A violation means an article was assigned to a period that closed BEFORE the
article was published -- i.e. the join layer leaked future information into
that period's feature row. The guard compares both timestamps in UTC (the
price-side ``period_close_ts`` is stored tz-aware ``America/New_York`` and is
converted to UTC for the comparison; ``published_at`` is already UTC). The
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

import pandas as pd

from equity.data.registry import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics" / "equity"


def run_published_at_guard(joined_df: pd.DataFrame) -> dict[str, Any]:
    """Assert ``published_at <= period_close_ts`` for every row of ``joined_df``.

    Both timestamps are compared in UTC. ``period_close_ts`` may be tz-aware
    ``America/New_York`` (the canonical price-side dtype) or already UTC; it is
    coerced to UTC. ``published_at`` is expected to be tz-aware UTC (the
    canonical articles dtype); a tz-naive column raises ``ValueError`` (it
    indicates a schema slip upstream).

    Parameters
    ----------
    joined_df:
        DataFrame with at least the columns ``ticker``, ``period_close_ts``,
        ``published_at``, ``text``, ``source`` -- the output shape of
        :meth:`equity.data.loader.EquityDataLoader.join_articles_to_prices`.

    Returns
    -------
    dict
        ``{"pass": bool, "n_violations": int, "violations": list[dict],
        "n_checked": int}``. Each violation dict carries the row's
        ``ticker``, ``published_at`` (ISO UTC), ``period_close_ts`` (ISO UTC)
        and the gap in seconds (positive => leak).
    """
    required = {"ticker", "period_close_ts", "published_at", "text", "source"}
    missing = required - set(joined_df.columns)
    if missing:
        raise ValueError(f"joined_df missing required columns: {sorted(missing)}")

    if joined_df.empty:
        return {
            "pass": True,
            "n_violations": 0,
            "violations": [],
            "n_checked": 0,
        }

    pc = pd.Series(joined_df["period_close_ts"]).copy()
    if pc.dt.tz is None:
        # Assume America/New_York if tz-naive (price-side canonical tz); this
        # is a defensive convenience. A truly tz-naive column would indicate a
        # schema slip upstream that we surface here rather than silently mask.
        raise ValueError(
            "period_close_ts is tz-naive in the joined frame; expected "
            "tz-aware America/New_York (or UTC). Refusing to compare."
        )
    pc_utc = pc.dt.tz_convert("UTC")

    pub = pd.Series(joined_df["published_at"]).copy()
    if pub.dt.tz is None:
        raise ValueError(
            "published_at is tz-naive in the joined frame; expected "
            "tz-aware UTC. Refusing to compare (potential point-in-time leak)."
        )
    pub_utc = pub.dt.tz_convert("UTC")

    # Violation: published_at > period_close_ts (article leaks into a period
    # that closed before publication).
    gap = pub_utc.to_numpy() - pc_utc.to_numpy()  # timedelta64[ns]
    violation_mask = gap > pd.Timedelta(0)

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
    }


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
            "provided, --prices/--articles are ignored."
        ),
    )
    parser.add_argument(
        "--prices",
        help="Path to the partitioned prices parquet store (used when --joined is absent).",
    )
    parser.add_argument(
        "--articles",
        help="Path to the partitioned articles parquet store (used when --joined is absent).",
    )
    parser.add_argument(
        "--output",
        help=(
            "Optional path to write the JSON result. When omitted, the result "
            f"is written under {RESULTS_DIR} with a UTC timestamp suffix."
        ),
    )
    args = parser.parse_args()

    if args.joined:
        joined = pd.read_parquet(args.joined)
    else:
        if not args.prices or not args.articles:
            parser.error(
                "Either --joined OR both --prices and --articles are required."
            )
        # The join needs an EquityDataLoader with a universe + window. We
        # default to the sp500 universe and the loader's default window
        # inferred from the prices frame's [min, max] period_close_ts. This
        # mirrors how a user would invoke the guard on a freshly-built store.
        from equity.data.loader import EquityDataLoader

        prices = pd.read_parquet(args.prices)
        start = pd.Timestamp(prices["period_close_ts"].min()).normalize()
        end = pd.Timestamp(prices["period_close_ts"].max()).normalize() + pd.Timedelta(days=1)
        loader = EquityDataLoader("sp500", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        joined_path = loader.join_articles_to_prices(
            prices_path=args.prices,
            articles_path=args.articles,
        )
        joined = pd.read_parquet(joined_path)

    result = run_published_at_guard(joined)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"published_at_guard_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nResult written to: {out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
