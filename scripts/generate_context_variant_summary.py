#!/usr/bin/env python3
"""
@module: scripts.generate_context_variant_summary
@depends: pandas
@exports: generate_context_variant_summary
@data_flow: context_variant sweep csvs -> combined rebuttal summary artifacts

Generate a cross-model summary from multiple context-variant sweep directories.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS_DIRS = [
    PROJECT_ROOT / "results" / "context_variant_matrix_20260330_093315",
    PROJECT_ROOT / "results" / "context_variant_matrix_extra_trees_20260330_094030",
    PROJECT_ROOT / "results" / "context_variant_matrix_catboost_20260330_094304",
]

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "extra_trees": "Extra Trees",
    "catboost": "CatBoost",
}

VARIANT_LABELS = {
    "sce": "SCE",
    "target_mean": "Target Mean",
    "hierarchical_mean_count": "Hierarchical Mean+Count",
    "hierarchical_mean_std_count": "Hierarchical Mean+Std+Count",
}


def _label(value: str, mapping: dict[str, str]) -> str:
    return mapping.get(value, value.replace("_", " ").title())


def _load_results(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "all_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find results file: {csv_path}")

    df = pd.read_csv(csv_path, decimal=",")
    numeric_cols = [
        "baseline_rmse",
        "baseline_r2",
        "enriched_rmse",
        "enriched_r2",
        "rmse_improvement_pct",
        "r2_improvement_pct",
        "n_samples",
        "n_baseline_features",
        "n_enriched_features",
        "runtime_seconds",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["results_dir"] = results_dir.name
    return df


def _build_summary_markdown(
    overall_avg: pd.DataFrame,
    model_avg: pd.DataFrame,
    win_counts: pd.DataFrame,
    cell_winners: pd.DataFrame,
) -> str:
    overall_best = overall_avg.iloc[0]
    lines = [
        "# Cross-Model Context Variant Summary",
        "",
        "## Headline",
        "",
        (
            f"Across all evaluated model-dataset pairs, `{overall_best['context_variant_label']}` "
            f"has the strongest mean RMSE improvement at `{overall_best['mean_rmse_improvement_pct']:+.2f}%`."
        ),
        "",
        "## Average RMSE Improvement By Model",
        "",
        "| Model | Context Variant | Mean RMSE Improvement % | Mean R2 Improvement % | Mean Runtime (s) | Mean Added Features |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for _, row in model_avg.iterrows():
        lines.append(
            "| {model_type_label} | {context_variant_label} | {mean_rmse_improvement_pct:+.2f} | {mean_r2_improvement_pct:+.2f} | {mean_runtime_seconds:.2f} | {mean_added_features:.1f} |".format(
                **row.to_dict()
            )
        )

    lines.extend(
        [
            "",
            "## Variant Win Counts",
            "",
            "Counts below are based on the best RMSE improvement inside each model-dataset cell.",
            "",
            "| Context Variant | Win Count | Mean Winning Improvement % |",
            "|---|---:|---:|",
        ]
    )

    for _, row in win_counts.iterrows():
        lines.append(
            "| {context_variant_label} | {win_count} | {mean_winning_improvement_pct:+.2f} |".format(
                **row.to_dict()
            )
        )

    lines.extend(
        [
            "",
            "## Per Model-Dataset Winner",
            "",
            "| Model | Dataset | Winning Variant | RMSE Improvement % | Baseline RMSE | Enriched RMSE |",
            "|---|---|---|---:|---:|---:|",
        ]
    )

    for _, row in cell_winners.iterrows():
        lines.append(
            "| {model_type_label} | {dataset} | {context_variant_label} | {rmse_improvement_pct:+.2f} | {baseline_rmse:,.2f} | {enriched_rmse:,.2f} |".format(
                **row.to_dict()
            )
        )

    return "\n".join(lines) + "\n"


def generate_context_variant_summary(results_dirs: list[Path], output_dir: Path) -> Path:
    frames = [_load_results(path) for path in results_dirs]
    combined = pd.concat(frames, ignore_index=True)
    combined["model_type_label"] = combined["model_type"].map(lambda value: _label(value, MODEL_LABELS))
    combined["context_variant_label"] = combined["context_variant"].map(lambda value: _label(value, VARIANT_LABELS))
    combined["added_features"] = combined["n_enriched_features"] - combined["n_baseline_features"]

    output_dir.mkdir(parents=True, exist_ok=True)
    combined.sort_values(["model_type", "dataset", "context_variant"]).to_csv(
        output_dir / "combined_results.csv", index=False
    )

    model_avg = (
        combined.groupby(["model_type", "model_type_label", "context_variant", "context_variant_label"], as_index=False)
        .agg(
            mean_rmse_improvement_pct=("rmse_improvement_pct", "mean"),
            mean_r2_improvement_pct=("r2_improvement_pct", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_added_features=("added_features", "mean"),
        )
        .sort_values(["model_type_label", "mean_rmse_improvement_pct"], ascending=[True, False])
    )
    model_avg.to_csv(output_dir / "model_variant_averages.csv", index=False)

    overall_avg = (
        combined.groupby(["context_variant", "context_variant_label"], as_index=False)
        .agg(
            mean_rmse_improvement_pct=("rmse_improvement_pct", "mean"),
            mean_r2_improvement_pct=("r2_improvement_pct", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            mean_added_features=("added_features", "mean"),
        )
        .sort_values("mean_rmse_improvement_pct", ascending=False)
    )
    overall_avg.to_csv(output_dir / "overall_variant_averages.csv", index=False)

    per_model_best = (
        model_avg.groupby(["model_type", "model_type_label"], as_index=False)
        .first()
        .sort_values("model_type_label")
    )
    per_model_best.to_csv(output_dir / "best_variant_by_model.csv", index=False)

    cell_winners = (
        combined.sort_values(
            ["model_type", "dataset", "rmse_improvement_pct", "enriched_rmse"],
            ascending=[True, True, False, True],
        )
        .groupby(["model_type", "dataset"], as_index=False)
        .first()
        .sort_values(["model_type_label", "dataset"])
    )
    cell_winners.to_csv(output_dir / "best_variant_by_model_dataset.csv", index=False)

    win_counts = (
        cell_winners.groupby(["context_variant", "context_variant_label"], as_index=False)
        .agg(
            win_count=("dataset", "count"),
            mean_winning_improvement_pct=("rmse_improvement_pct", "mean"),
        )
        .sort_values(["win_count", "mean_winning_improvement_pct"], ascending=[False, False])
    )
    win_counts.to_csv(output_dir / "variant_win_counts.csv", index=False)

    summary_md = _build_summary_markdown(overall_avg, model_avg, win_counts, cell_winners)
    (output_dir / "summary.md").write_text(summary_md, encoding="utf-8")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a cross-model context variant summary")
    parser.add_argument(
        "--results-dir",
        dest="results_dirs",
        action="append",
        type=Path,
        help="Context-variant sweep directory containing all_results.csv. Repeat for multiple directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where combined summary artifacts will be written.",
    )
    args = parser.parse_args()

    results_dirs = args.results_dirs or DEFAULT_RESULTS_DIRS
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "results" / f"context_variant_cross_model_{timestamp}"

    generated_dir = generate_context_variant_summary(results_dirs, output_dir)
    print(f"Generated cross-model context summary in: {generated_dir}")


if __name__ == "__main__":
    main()