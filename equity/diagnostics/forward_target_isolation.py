"""
@module: equity.diagnostics.forward_target_isolation
@depends: pandas
@exports: run_forward_target_isolation, main
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §9.3 (FOOTGUN)
@data_flow: feature matrix -> scan for ret_hN / pred_hN leaks -> JSON report

S5.3 forward-target isolation diagnostic. Asserts that none of the forward
labels (``ret_hN``) or final predictions (``pred_hN``) appear in a feature
matrix handed to a forecaster layer. Residual-layer injection of OOF
``pred_sector_hN`` is allow-listed via ``allowed_pred_sector=True``.

CLI:

    python -m equity.diagnostics.forward_target_isolation \\
        --features <features.parquet> [--output <report.json>] [--strict]

Exits 0 on PASS, 1 on any violation. Mirrors
:mod:`equity.diagnostics.lookahead_indicator` CLI shape.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics" / "equity"

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 21, 63)

# ret_hN / pred_hN / pred_hN_qXX / resid_hN / pred_resid_hN
_RET_H_RE = re.compile(r"^ret_h(\d+)$")
_PRED_H_RE = re.compile(r"^pred_h(\d+)(_q\d+)?$")
_PRED_RESID_RE = re.compile(r"^pred_resid_h(\d+)$")
_RESID_H_RE = re.compile(r"^resid_h(\d+)$")
_PRED_SECTOR_RE = re.compile(r"^pred_sector_h(\d+)$")


def _resolve_under_project_root(path: str | Path, project_root: Path | None = None) -> Path:
    """Resolve ``--output`` safely. Mirrors lookahead_indicator containment."""
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


def _horizon_set(horizons: Sequence[int]) -> set[int]:
    return {int(h) for h in horizons}


def run_forward_target_isolation(
    features: pd.DataFrame,
    *,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    allowed_pred_sector: bool = False,
    extra_forbidden: Optional[Iterable[str]] = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Scan ``features`` for forward-target / prediction leakage columns.

    Parameters
    ----------
    features:
        Candidate feature matrix (columns inspected). Does not need to carry
        data — only column names matter.
    horizons:
        Horizons to flag (default Q5 lock-in ``{1,5,10,21,63}``).
    allowed_pred_sector:
        When True, ``pred_sector_hN`` columns are permitted (residual layer
        design matrix). Default False — a feature matrix that still carries
        sector-head preds is treated as a leak for the *sector* head / SCE.
    extra_forbidden:
        Additional exact column names to flag.
    strict:
        Reserved for CLI symmetry with other guards (currently unused beyond
        being recorded in the result dict).

    Returns
    -------
    dict
        ``{pass, n_violations, violations, forbidden_found, horizons, ...}``.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("run_forward_target_isolation: features must be a DataFrame")

    hs = _horizon_set(horizons)
    violations: list[dict[str, Any]] = []
    forbidden_found: list[str] = []
    extra = set(extra_forbidden or ())

    for col in features.columns:
        if not isinstance(col, str):
            continue
        if col in extra:
            violations.append(
                {
                    "type": "extra_forbidden",
                    "column": col,
                    "reason": "listed in extra_forbidden",
                }
            )
            forbidden_found.append(col)
            continue

        m = _RET_H_RE.match(col)
        if m is not None:
            h = int(m.group(1))
            if h in hs or not hs:
                violations.append(
                    {
                        "type": "forward_target_in_features",
                        "column": col,
                        "horizon": h,
                        "reason": (
                            "ret_hN is a LABEL only — never a feature "
                            "(PRD §9.3 / FOOTGUN forward-target leak)"
                        ),
                    }
                )
                forbidden_found.append(col)
                continue

        m = _PRED_H_RE.match(col)
        if m is not None:
            h = int(m.group(1))
            if h in hs or not hs:
                violations.append(
                    {
                        "type": "prediction_in_features",
                        "column": col,
                        "horizon": h,
                        "reason": "pred_hN (final or quantile) must not be a feature input",
                    }
                )
                forbidden_found.append(col)
                continue

        m = _PRED_RESID_RE.match(col) or _RESID_H_RE.match(col)
        if m is not None:
            h = int(m.group(1))
            if h in hs or not hs:
                violations.append(
                    {
                        "type": "residual_in_features",
                        "column": col,
                        "horizon": h,
                        "reason": "resid_hN / pred_resid_hN must not be design-matrix features",
                    }
                )
                forbidden_found.append(col)
                continue

        m = _PRED_SECTOR_RE.match(col)
        if m is not None and not allowed_pred_sector:
            h = int(m.group(1))
            if h in hs or not hs:
                violations.append(
                    {
                        "type": "sector_pred_in_features",
                        "column": col,
                        "horizon": h,
                        "reason": (
                            "pred_sector_hN only allowed as OOF residual-layer "
                            "feature (pass allowed_pred_sector=True)"
                        ),
                    }
                )
                forbidden_found.append(col)

    result = {
        "pass": len(violations) == 0,
        "n_violations": len(violations),
        "violations": violations,
        "forbidden_found": sorted(set(forbidden_found)),
        "horizons": sorted(hs),
        "allowed_pred_sector": bool(allowed_pred_sector),
        "n_columns_scanned": int(features.shape[1]),
        "strict": bool(strict),
    }
    return result


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Forward-target isolation guard (S5.3): asserts no ret_hN / pred_hN "
            "columns appear in the feature matrix. Exits 0 on PASS, 1 on leak."
        )
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Path to the candidate feature matrix parquet.",
    )
    parser.add_argument(
        "--horizons",
        default="1,5,10,21,63",
        help="Comma-separated horizons to flag (default 1,5,10,21,63).",
    )
    parser.add_argument(
        "--allow-pred-sector",
        action="store_true",
        help="Permit pred_sector_hN columns (residual-layer design matrix).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON result (resolved under PROJECT_ROOT).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict flag (recorded in the report; reserved for future checks).",
    )
    args = parser.parse_args(argv)

    features = pd.read_parquet(args.features)
    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    result = run_forward_target_isolation(
        features,
        horizons=horizons,
        allowed_pred_sector=bool(args.allow_pred_sector),
        strict=bool(args.strict),
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = _resolve_under_project_root(args.output)
    else:
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"forward_target_isolation_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nResult written to: {out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
