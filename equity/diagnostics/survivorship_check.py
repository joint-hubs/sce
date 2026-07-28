"""
@module: equity.diagnostics.survivorship_check
@depends: pandas
@exports: run_survivorship_check, main
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §8.1 (survivorship)
@data_flow: universe name|path -> configs/equity/<name>_universe.csv (or path)
            -> count non-null delisted_at -> pass if n_delisted >= min_delisted

Survivorship-bias seed check (S4.5). A training universe that only carries
still-listed names is survivorship-biased; the equity seed universe must keep
enough delisted tickers for PIT backtests to see bankruptcies / acquisitions.

Resolution rules for ``universe``:

* bare name (e.g. ``"sp500"``) → ``configs/equity/sp500_universe.csv``;
* path ending in ``.csv`` / ``.parquet`` → read that file directly.

CSV columns: ``ticker, listed_at, delisted_at, name`` (``#`` comment lines ok).
``pass = n_delisted >= min_delisted`` (default 13 — the S1 seed has 13 delisted).

CLI:

    python -m equity.diagnostics.survivorship_check \\
        --universe sp500 [--min-delisted 13] [--output <report.json>]

Exits 0 on PASS, 1 on fail. Mirrors the
:mod:`equity.diagnostics.lookahead_indicator` CLI shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics" / "equity"
CONFIGS_EQUITY_DIR = PROJECT_ROOT / "configs" / "equity"

_REQUIRED_COLS = ("ticker", "delisted_at")


def _resolve_under_project_root(path: str | Path, project_root: Path | None = None) -> Path:
    """Resolve ``--output`` safely. Refuses ``..`` traversal AND absolute paths
    outside ``project_root``. Mirrors
    :func:`equity.diagnostics.lookahead_indicator._resolve_under_project_root`.
    """
    root = project_root if project_root is not None else PROJECT_ROOT
    raw = Path(path)
    parts = list(raw.parts)
    if ".." in parts:
        raise ValueError(f"Refusing --output with path-traversal component '..': {path}.")
    if not raw.is_absolute():
        resolved = (root / raw).resolve()
    else:
        resolved = raw.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Refusing --output outside project root ({root}): {path}.") from exc
    return resolved


def _resolve_universe_path(universe: str | Path) -> Path:
    """Map a universe name or path to an on-disk CSV/parquet."""
    raw = Path(universe)
    # Bare name (no suffix, not an existing path) -> configs/equity/<name>_universe.csv
    if raw.suffix.lower() in {".csv", ".parquet"}:
        path = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
        if not path.is_file():
            raise FileNotFoundError(f"Universe file not found: {path}")
        return path

    name = str(universe).strip()
    if not name or "/" in name or "\\" in name or name.startswith("."):
        # Looks like a path without a recognised suffix, or empty.
        candidate = raw if raw.is_absolute() else (PROJECT_ROOT / raw)
        if candidate.is_file():
            return candidate
        raise ValueError(
            f"Unknown universe {universe!r}: expected a bare name "
            f"(e.g. 'sp500' → configs/equity/sp500_universe.csv) or a path to a "
            f".csv/.parquet file."
        )

    path = CONFIGS_EQUITY_DIR / f"{name}_universe.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Universe {name!r} not found at {path}. "
            f"Available configs live under {CONFIGS_EQUITY_DIR}."
        )
    return path


def _read_universe(path: Path) -> pd.DataFrame:
    """Load a universe table from CSV or parquet."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    elif suffix == ".csv":
        df = pd.read_csv(path, comment="#")
    else:
        raise ValueError(f"Unsupported universe file type: {path.suffix}")
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Universe file {path} missing required columns {missing}; have {list(df.columns)}"
        )
    return df


def run_survivorship_check(
    universe: str | Path,
    *,
    min_delisted: int = 13,
) -> dict[str, Any]:
    """Count delisted tickers in a universe and gate on ``min_delisted``.

    Parameters
    ----------
    universe:
        Bare name (``"sp500"`` → ``configs/equity/sp500_universe.csv``) or a
        path to a CSV/parquet with columns ``ticker, listed_at, delisted_at,
        name``.
    min_delisted:
        Minimum number of rows with a non-null ``delisted_at`` required to
        pass (default 13 — the S1 seed has exactly 13 delisted names).

    Returns
    -------
    dict
        ``{pass, n_delisted, n_required, n_total, missing_tickers, universe}``.
        ``missing_tickers`` is reserved for future named-delisted audits and is
        currently always an empty list (count-only check).
    """
    if min_delisted < 0:
        raise ValueError(f"min_delisted must be >= 0; got {min_delisted}")

    path = _resolve_universe_path(universe)
    df = _read_universe(path)

    # Count only values that parse as dates. The seed CSV writes living tickers
    # as ``ticker,,name`` (3 fields) so pandas shifts the name into
    # ``delisted_at``; a pure notna()/non-empty check would false-positive every
    # living name. ``errors='coerce'`` turns non-dates into NaT.
    delisted_ts = pd.to_datetime(df["delisted_at"], errors="coerce", utc=True)
    delisted_mask = delisted_ts.notna()
    n_delisted = int(delisted_mask.sum())
    n_total = int(len(df))

    return {
        "pass": n_delisted >= int(min_delisted),
        "n_delisted": n_delisted,
        "n_required": int(min_delisted),
        "n_total": n_total,
        "missing_tickers": [],
        "universe": str(universe),
        "universe_path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Survivorship-bias seed check (S4.5): counts non-null delisted_at "
            "rows in a universe CSV and passes when n_delisted >= min_delisted. "
            "Exits 0 on PASS, 1 on fail."
        ),
    )
    parser.add_argument(
        "--universe",
        required=True,
        help=(
            "Bare universe name (e.g. 'sp500' → configs/equity/sp500_universe.csv) "
            "or a path to a ticker CSV/parquet."
        ),
    )
    parser.add_argument(
        "--min-delisted",
        type=int,
        default=13,
        help="Minimum number of delisted tickers required to pass (default 13).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON result (resolved under PROJECT_ROOT).",
    )
    args = parser.parse_args()

    result = run_survivorship_check(args.universe, min_delisted=args.min_delisted)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = _resolve_under_project_root(args.output)
    else:
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"survivorship_check_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nResult written to: {out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
