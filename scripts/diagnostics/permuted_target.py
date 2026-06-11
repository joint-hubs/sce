"""Permuted-target leakage diagnostic."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scripts.diagnostics._common import evaluate_config_dataframe, load_config_and_dataset

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "diagnostics"
DEFAULT_MAX_ROWS = 20000


def _resolve_effective_max_rows(max_rows: int | None, run_grade: str) -> int | None:
    if run_grade == "report-grade":
        return None
    return max_rows


def run_permuted_target(
    config_name: str,
    n_permutations: int = 5,
    seed: int = 42,
    max_rows: int | None = DEFAULT_MAX_ROWS,
    run_grade: str = "diagnostic",
) -> dict:
    config, _, df, target_col = load_config_and_dataset(config_name)
    source_rows = len(df)
    effective_max_rows = _resolve_effective_max_rows(max_rows, run_grade)
    if effective_max_rows is not None and len(df) > effective_max_rows:
        df = df.sample(n=effective_max_rows, random_state=seed)

    real_metrics = evaluate_config_dataframe(config, config_name, df, target_col)
    real_advantage = ((real_metrics["baseline_rmse"] - real_metrics["sce_rmse"]) / real_metrics["baseline_rmse"]) * 100

    rng = np.random.default_rng(seed)
    permuted_advantages: list[float] = []
    baseline_perm: list[float] = []
    sce_perm: list[float] = []
    values = df[target_col].to_numpy(copy=True)
    for _ in range(n_permutations):
        perm_df = df.copy()
        perm_df[target_col] = rng.permutation(values)
        metrics = evaluate_config_dataframe(config, config_name, perm_df, target_col)
        baseline_perm.append(metrics["baseline_rmse"])
        sce_perm.append(metrics["sce_rmse"])
        advantage = ((metrics["baseline_rmse"] - metrics["sce_rmse"]) / metrics["baseline_rmse"]) * 100
        permuted_advantages.append(float(advantage))

    result = {
        "run_grade": run_grade,
        "source_rows": int(source_rows),
        "evaluated_rows": int(len(df)),
        "subsample_max_rows": effective_max_rows,
        "baseline_rmse_real": float(real_metrics["baseline_rmse"]),
        "sce_rmse_real": float(real_metrics["sce_rmse"]),
        "baseline_rmse_permuted_mean": float(np.mean(baseline_perm)) if baseline_perm else 0.0,
        "sce_rmse_permuted_mean": float(np.mean(sce_perm)) if sce_perm else 0.0,
        "sce_advantage_real": float(real_advantage),
        "sce_advantage_permuted_mean": float(np.mean(permuted_advantages)) if permuted_advantages else 0.0,
        "permuted_advantages": permuted_advantages,
        "pass": float(np.mean(permuted_advantages)) < 1.0 if permuted_advantages else True,
    }

    out_dir = RESULTS_DIR / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"permuted_target_{ts}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run permuted-target leakage diagnostic")
    parser.add_argument("--dataset", required=True, help="Dataset config name")
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS, help="Subsample size for diagnostics")
    parser.add_argument("--full", action="store_true", help="Use full dataset without subsampling")
    parser.add_argument(
        "--run-grade",
        choices=["exploratory", "diagnostic", "report-grade"],
        default="diagnostic",
        help="report-grade forces full-dataset diagnostics",
    )
    args = parser.parse_args()

    result = run_permuted_target(
        args.dataset,
        n_permutations=args.n_permutations,
        seed=args.seed,
        max_rows=None if args.full else args.max_rows,
        run_grade=args.run_grade,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
