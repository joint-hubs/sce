"""
@module: equity.diagnostics.walk_forward_monotonicity
@depends: pandas
@exports: run_walk_forward_monotonicity, main, normalize_folds
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §8.1 (PIT folds)
@data_flow: fold list (explicit or SCE _last_fold_timestamps) -> per-fold
            train_max/val_min(/test) ordering checks -> JSON report

Walk-forward monotonicity diagnostic (S4.5). Asserts that each fold's train
window ends at or before the validation window starts (``train_max <= val_min``;
``--strict`` tightens to ``train_max < val_min``) and, when test bounds are
present, that validation ends at or before the test window starts
(``val_max <= test_min`` / strict ``<``).

Accepts either an explicit fold list::

    {train_start?, train_max, val_min, val_max, test_min?, test_max?}

or the SCE engine's ``_last_fold_timestamps`` shape::

    {train_size, val_size, train_max, val_min, val_max}

so callers may pass ``enricher._last_fold_timestamps`` directly. Gap=0 is
allowed under the default (non-strict) comparison per PRD §8.1.

CLI:

    python -m equity.diagnostics.walk_forward_monotonicity \\
        --folds <json-path-or-inline> [--strict] [--output <report.json>]

Exits 0 on PASS, 1 on any violation. Mirrors the
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

# Keys that mark an SCE-style fold record (engine._last_fold_timestamps).
_SCE_FOLD_KEYS = frozenset({"train_max", "val_min", "val_max"})


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


def _to_ts(value: Any) -> pd.Timestamp:
    """Coerce a fold bound to a comparable :class:`pandas.Timestamp`."""
    if isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value)


def normalize_folds(folds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize fold records to a common shape.

    Accepts:

    * explicit folds with ``train_max`` / ``val_min`` / ``val_max`` (+ optional
      ``train_start``, ``test_min``, ``test_max``);
    * SCE ``_last_fold_timestamps`` records
      (``{train_size, val_size, train_max, val_min, val_max}``) — test bounds
      are simply absent.

    Returns a list of dicts with Timestamp-coerced bounds. Extra keys (e.g.
    ``train_size``) are preserved for diagnostics but not required.
    """
    if not isinstance(folds, list):
        raise TypeError(f"folds must be a list of dicts; got {type(folds).__name__}")

    out: list[dict[str, Any]] = []
    for i, raw in enumerate(folds):
        if not isinstance(raw, dict):
            raise TypeError(f"fold[{i}] must be a dict; got {type(raw).__name__}")
        missing = _SCE_FOLD_KEYS - set(raw)
        if missing:
            raise ValueError(
                f"fold[{i}] missing required keys {sorted(missing)}; have {sorted(raw)}"
            )
        fold: dict[str, Any] = dict(raw)
        for key in ("train_start", "train_max", "val_min", "val_max", "test_min", "test_max"):
            if key in fold and fold[key] is not None:
                fold[key] = _to_ts(fold[key])
        out.append(fold)
    return out


def run_walk_forward_monotonicity(
    folds: list[dict[str, Any]],
    *,
    strict: bool = False,
) -> dict[str, Any]:
    """Check walk-forward fold monotonicity and return a result dict.

    Parameters
    ----------
    folds:
        Fold list — either explicit ``{train_max, val_min, val_max, ...}``
        dicts or SCE ``_last_fold_timestamps`` records (see
        :func:`normalize_folds`).
    strict:
        When True, require strict inequalities (``train_max < val_min``,
        ``val_max < test_min``). Default allows equality (gap=0 per PRD §8.1).

    Returns
    -------
    dict
        ``{pass, n_folds, n_violations, violations, strict}``.
    """
    normalized = normalize_folds(folds)
    violations: list[dict[str, Any]] = []

    for i, fold in enumerate(normalized):
        train_max = fold["train_max"]
        val_min = fold["val_min"]
        val_max = fold["val_max"]

        if strict:
            if not (train_max < val_min):
                violations.append(
                    {
                        "fold_idx": i,
                        "reason": "train_max_not_before_val_min",
                        "train_max": str(train_max),
                        "val_min": str(val_min),
                        "strict": True,
                    }
                )
        else:
            if not (train_max <= val_min):
                violations.append(
                    {
                        "fold_idx": i,
                        "reason": "train_max_after_val_min",
                        "train_max": str(train_max),
                        "val_min": str(val_min),
                        "strict": False,
                    }
                )

        test_min = fold.get("test_min")
        if test_min is not None and val_max is not None:
            if strict:
                if not (val_max < test_min):
                    violations.append(
                        {
                            "fold_idx": i,
                            "reason": "val_max_not_before_test_min",
                            "val_max": str(val_max),
                            "test_min": str(test_min),
                            "strict": True,
                        }
                    )
            else:
                if not (val_max <= test_min):
                    violations.append(
                        {
                            "fold_idx": i,
                            "reason": "val_max_after_test_min",
                            "val_max": str(val_max),
                            "test_min": str(test_min),
                            "strict": False,
                        }
                    )

    n_violations = len(violations)
    return {
        "pass": n_violations == 0,
        "n_folds": len(normalized),
        "n_violations": n_violations,
        "violations": violations,
        "strict": bool(strict),
    }


def _load_folds(raw: str) -> list[dict[str, Any]]:
    """Load folds from a filesystem path or an inline JSON array/object string."""
    path = Path(raw)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        payload = json.loads(text)
    else:
        payload = json.loads(raw)

    if isinstance(payload, dict) and "folds" in payload:
        payload = payload["folds"]
    if not isinstance(payload, list):
        raise ValueError(
            f"--folds must be a JSON list (or {{'folds': [...]}}); got {type(payload).__name__}"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Walk-forward monotonicity guard (S4.5): asserts train_max <= val_min "
            "(and val_max <= test_min when present) for each fold. Accepts SCE "
            "_last_fold_timestamps shape. Exits 0 on PASS, 1 on any violation."
        ),
    )
    parser.add_argument(
        "--folds",
        required=True,
        help=(
            "Path to a JSON file of folds, OR an inline JSON list of fold dicts "
            '(e.g. \'[{"train_max":"2020-01-01","val_min":"2020-01-02","val_max":"2020-03-01"}]\').'
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON result (resolved under PROJECT_ROOT).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Require strict inequalities (train_max < val_min; val_max < test_min).",
    )
    args = parser.parse_args()

    folds = _load_folds(args.folds)
    result = run_walk_forward_monotonicity(folds, strict=args.strict)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = _resolve_under_project_root(args.output)
    else:
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"walk_forward_monotonicity_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    print(json.dumps(result, indent=2, default=str))
    print(f"\nResult written to: {out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
