"""Cross-fit on/off A/B diagnostic."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.diagnostics._common import evaluate_config_dataframe, load_config_and_dataset

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "diagnostics"
DEFAULT_MAX_ROWS = 20000


def _resolve_effective_max_rows(max_rows: int | None, run_grade: str) -> int | None:
    if run_grade == "report-grade":
        return None
    return max_rows


def run_crossfit_ab(
    config_name: str,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    seed: int = 42,
    run_grade: str = "diagnostic",
) -> dict:
    config, _, df, target_col = load_config_and_dataset(config_name)
    source_rows = len(df)
    effective_max_rows = _resolve_effective_max_rows(max_rows, run_grade)
    if effective_max_rows is not None and len(df) > effective_max_rows:
        df = df.sample(n=effective_max_rows, random_state=seed)

    with_cf = evaluate_config_dataframe(config, config_name, df, target_col, use_cross_fitting_override=True)
    without_cf = evaluate_config_dataframe(config, config_name, df, target_col, use_cross_fitting_override=False)

    rmse_cf = float(with_cf["sce_rmse"])
    rmse_no_cf = float(without_cf["sce_rmse"])
    leakage_signal_pp = ((rmse_no_cf - rmse_cf) / rmse_no_cf) * 100 if rmse_no_cf else 0.0

    result = {
        "run_grade": run_grade,
        "source_rows": int(source_rows),
        "evaluated_rows": int(len(df)),
        "subsample_max_rows": effective_max_rows,
        "rmse_cf": rmse_cf,
        "rmse_no_cf": rmse_no_cf,
        "r2_cf": float(with_cf.get("sce_r2", 0.0)),
        "r2_no_cf": float(without_cf.get("sce_r2", 0.0)),
        "leakage_signal_pp": float(leakage_signal_pp),
    }

    out_dir = RESULTS_DIR / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"crossfit_ab_{ts}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run cross-fit A/B diagnostic")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS, help="Subsample size for diagnostics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full", action="store_true", help="Use full dataset without subsampling")
    parser.add_argument(
        "--run-grade",
        choices=["exploratory", "diagnostic", "report-grade"],
        default="diagnostic",
        help="report-grade forces full-dataset diagnostics",
    )
    args = parser.parse_args()

    result = run_crossfit_ab(
        args.dataset,
        max_rows=None if args.full else args.max_rows,
        seed=args.seed,
        run_grade=args.run_grade,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
