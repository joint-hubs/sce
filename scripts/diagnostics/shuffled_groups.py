"""Shuffled-groups structure diagnostic."""

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


def run_shuffled_groups(
    config_name: str,
    n_permutations: int = 5,
    seed: int = 42,
    columns: list[str] | None = None,
    mode: str = "all",
    max_rows: int | None = DEFAULT_MAX_ROWS,
    run_grade: str = "diagnostic",
) -> dict:
    config, _, df, target_col = load_config_and_dataset(config_name)
    source_rows = len(df)
    effective_max_rows = _resolve_effective_max_rows(max_rows, run_grade)
    if effective_max_rows is not None and len(df) > effective_max_rows:
        df = df.sample(n=effective_max_rows, random_state=seed)

    categorical_cols = columns or config.get("features", {}).get("categorical", [])
    available_cols = [col for col in categorical_cols if col in df.columns]
    if not available_cols:
        raise ValueError("No categorical columns available for shuffled-groups diagnostic")

    real_metrics = evaluate_config_dataframe(config, config_name, df, target_col)
    real_advantage = ((real_metrics["baseline_rmse"] - real_metrics["sce_rmse"]) / real_metrics["baseline_rmse"]) * 100

    rng = np.random.default_rng(seed)
    shuffled_advantages: list[float] = []
    per_column_advantages: dict[str, list[float]] = {}

    if mode == "all":
        for _ in range(n_permutations):
            shuffled_df = df.copy()
            for col in available_cols:
                shuffled_df[col] = shuffled_df[col].sample(
                    frac=1.0,
                    random_state=int(rng.integers(0, 1_000_000)),
                ).to_numpy()

            metrics = evaluate_config_dataframe(config, config_name, shuffled_df, target_col)
            advantage = ((metrics["baseline_rmse"] - metrics["sce_rmse"]) / metrics["baseline_rmse"]) * 100
            shuffled_advantages.append(float(advantage))
    elif mode == "per-column":
        for col in available_cols:
            col_advantages: list[float] = []
            for _ in range(n_permutations):
                shuffled_df = df.copy()
                shuffled_df[col] = shuffled_df[col].sample(
                    frac=1.0,
                    random_state=int(rng.integers(0, 1_000_000)),
                ).to_numpy()
                metrics = evaluate_config_dataframe(config, config_name, shuffled_df, target_col)
                advantage = ((metrics["baseline_rmse"] - metrics["sce_rmse"]) / metrics["baseline_rmse"]) * 100
                col_advantages.append(float(advantage))
            per_column_advantages[col] = col_advantages
            shuffled_advantages.extend(col_advantages)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    shuffled_mean = float(np.mean(shuffled_advantages)) if shuffled_advantages else 0.0
    threshold = 0.5 * real_advantage
    per_column_summary = {
        col: {
            "mean_advantage": float(np.mean(vals)) if vals else 0.0,
            "advantages": vals,
        }
        for col, vals in per_column_advantages.items()
    }
    result = {
        "run_grade": run_grade,
        "source_rows": int(source_rows),
        "evaluated_rows": int(len(df)),
        "subsample_max_rows": effective_max_rows,
        "sce_advantage_real": float(real_advantage),
        "sce_advantage_shuffled_mean": shuffled_mean,
        "shuffled_advantages": shuffled_advantages,
        "columns_evaluated": available_cols,
        "per_column": per_column_summary,
        "mode": mode,
        "pass": (real_advantage - shuffled_mean) > threshold if real_advantage > 0 else False,
    }

    out_dir = RESULTS_DIR / config_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"shuffled_groups_{mode}_{ts}.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run shuffled-groups diagnostic")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["all", "per-column"], default="all")
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS, help="Subsample size for diagnostics")
    parser.add_argument("--full", action="store_true", help="Use full dataset without subsampling")
    parser.add_argument(
        "--run-grade",
        choices=["exploratory", "diagnostic", "report-grade"],
        default="diagnostic",
        help="report-grade forces full-dataset diagnostics",
    )
    args = parser.parse_args()

    result = run_shuffled_groups(
        args.dataset,
        n_permutations=args.n_permutations,
        seed=args.seed,
        mode=args.mode,
        max_rows=None if args.full else args.max_rows,
        run_grade=args.run_grade,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
