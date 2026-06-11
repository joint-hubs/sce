#!/usr/bin/env python3
"""
@module: scripts.generate_categorical_mode_batch_summary
@depends: pandas, matplotlib
@exports: generate_summary, main
@data_flow: categorical comparison run dirs -> combined summary tables and markdown

Summarize a batch of manual-vs-auto categorical grouping comparison runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    }
)

COLORS = {
    "auto": "#1f78b4",
    "manual": "#d95f02",
    "positive": "#1b9e77",
    "negative": "#b24c63",
}

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient Boosting",
}


def _load_comparison(run_dir: Path) -> pd.DataFrame:
    csv_path = run_dir / "data" / "categorical_mode_comparison.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing comparison file: {csv_path}")

    df = pd.read_csv(csv_path)
    df["run_dir"] = run_dir.name
    df["run_path"] = str(run_dir)
    return df


def _run_timestamp(run_dir_name: str) -> str:
    return run_dir_name.rsplit("_", 2)[-2] + run_dir_name.rsplit("_", 2)[-1]


def _build_row(group: pd.DataFrame) -> dict[str, object]:
    run_dir = group["run_dir"].iloc[0]
    dataset = group["dataset"].iloc[0]
    model_type = group["model_type"].iloc[0]
    context_variant = group["context_variant"].iloc[0]
    manual = group[group["categorical_mode"] == "manual"].iloc[0]
    auto = group[group["categorical_mode"] == "auto"].iloc[0]

    auto_rmse_delta = float(auto["sce_rmse"] - manual["sce_rmse"])
    auto_r2_delta = float(auto["sce_r2"] - manual["sce_r2"])
    auto_improvement_delta = float(auto["rmse_improvement_pct"] - manual["rmse_improvement_pct"])
    auto_feature_delta = int(auto["n_sce_features"] - manual["n_sce_features"])

    return {
        "dataset": dataset,
        "model_type": model_type,
        "model_label": MODEL_LABELS.get(model_type, model_type.replace("_", " ").title()),
        "context_variant": context_variant,
        "run_dir": run_dir,
        "baseline_rmse": float(manual["baseline_rmse"]),
        "baseline_r2": float(manual["baseline_r2"]),
        "manual_sce_rmse": float(manual["sce_rmse"]),
        "auto_sce_rmse": float(auto["sce_rmse"]),
        "auto_minus_manual_sce_rmse": auto_rmse_delta,
        "manual_sce_r2": float(manual["sce_r2"]),
        "auto_sce_r2": float(auto["sce_r2"]),
        "auto_minus_manual_sce_r2": auto_r2_delta,
        "manual_rmse_improvement_pct": float(manual["rmse_improvement_pct"]),
        "auto_rmse_improvement_pct": float(auto["rmse_improvement_pct"]),
        "auto_minus_manual_rmse_improvement_pct": auto_improvement_delta,
        "manual_r2_improvement_pct": float(manual["r2_improvement_pct"]),
        "auto_r2_improvement_pct": float(auto["r2_improvement_pct"]),
        "auto_minus_manual_r2_improvement_pct": float(auto["r2_improvement_pct"] - manual["r2_improvement_pct"]),
        "manual_sce_features": int(manual["n_sce_features"]),
        "auto_sce_features": int(auto["n_sce_features"]),
        "auto_minus_manual_sce_features": auto_feature_delta,
        "manual_runtime_seconds": float(manual["runtime_seconds"]),
        "auto_runtime_seconds": float(auto["runtime_seconds"]),
        "winner": "auto" if auto_rmse_delta < 0 else "manual" if auto_rmse_delta > 0 else "tie",
    }


def _write_markdown(summary_df: pd.DataFrame, output_dir: Path) -> None:
    winner_counts = summary_df["winner"].value_counts()
    dataset_avg = (
        summary_df.groupby("dataset", as_index=False)
        .agg(
            mean_auto_minus_manual_sce_rmse=("auto_minus_manual_sce_rmse", "mean"),
            mean_auto_minus_manual_rmse_improvement_pct=("auto_minus_manual_rmse_improvement_pct", "mean"),
            auto_wins=("winner", lambda values: int((values == "auto").sum())),
            total_runs=("winner", "count"),
        )
        .sort_values("mean_auto_minus_manual_rmse_improvement_pct", ascending=False)
    )
    model_avg = (
        summary_df.groupby(["model_type", "model_label"], as_index=False)
        .agg(
            mean_auto_minus_manual_sce_rmse=("auto_minus_manual_sce_rmse", "mean"),
            mean_auto_minus_manual_rmse_improvement_pct=("auto_minus_manual_rmse_improvement_pct", "mean"),
            auto_wins=("winner", lambda values: int((values == "auto").sum())),
            total_runs=("winner", "count"),
        )
        .sort_values("mean_auto_minus_manual_rmse_improvement_pct", ascending=False)
    )
    best_auto = summary_df.sort_values("auto_minus_manual_rmse_improvement_pct", ascending=False).iloc[0]
    worst_auto = summary_df.sort_values("auto_minus_manual_rmse_improvement_pct", ascending=True).iloc[0]

    lines = [
        "# Categorical Mode Batch Summary",
        "",
        f"Runs summarized: {len(summary_df)}",
        f"Auto wins: {int(winner_counts.get('auto', 0))}",
        f"Manual wins: {int(winner_counts.get('manual', 0))}",
        f"Ties: {int(winner_counts.get('tie', 0))}",
        "",
        "## Strongest Auto Win",
        "",
        (
            f"{best_auto['dataset']} with {best_auto['model_label']}: "
            f"auto-minus-manual RMSE improvement delta {best_auto['auto_minus_manual_rmse_improvement_pct']:+.2f} pp, "
            f"RMSE delta {best_auto['auto_minus_manual_sce_rmse']:+,.2f}."
        ),
        "",
        "## Strongest Manual Win",
        "",
        (
            f"{worst_auto['dataset']} with {worst_auto['model_label']}: "
            f"auto-minus-manual RMSE improvement delta {worst_auto['auto_minus_manual_rmse_improvement_pct']:+.2f} pp, "
            f"RMSE delta {worst_auto['auto_minus_manual_sce_rmse']:+,.2f}."
        ),
        "",
        "## By Dataset",
        "",
        "| Dataset | Mean Auto-Manual RMSE Delta | Mean Auto-Manual Improvement Delta (pp) | Auto Wins | Total Runs |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in dataset_avg.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['mean_auto_minus_manual_sce_rmse']:+,.2f} | {row['mean_auto_minus_manual_rmse_improvement_pct']:+.2f} | {int(row['auto_wins'])} | {int(row['total_runs'])} |"
        )

    lines.extend(
        [
            "",
            "## By Model",
            "",
            "| Model | Mean Auto-Manual RMSE Delta | Mean Auto-Manual Improvement Delta (pp) | Auto Wins | Total Runs |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for _, row in model_avg.iterrows():
        lines.append(
            f"| {row['model_label']} | {row['mean_auto_minus_manual_sce_rmse']:+,.2f} | {row['mean_auto_minus_manual_rmse_improvement_pct']:+.2f} | {int(row['auto_wins'])} | {int(row['total_runs'])} |"
        )

    lines.extend(
        [
            "",
            "## All Runs",
            "",
            "| Dataset | Model | Winner | Manual RMSE | Auto RMSE | Auto-Manual RMSE Delta | Manual Improvement % | Auto Improvement % | Auto-Manual Improvement Delta (pp) |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    display_df = summary_df.sort_values(["dataset", "model_label"])
    for _, row in display_df.iterrows():
        lines.append(
            f"| {row['dataset']} | {row['model_label']} | {row['winner']} | {row['manual_sce_rmse']:,.2f} | {row['auto_sce_rmse']:,.2f} | {row['auto_minus_manual_sce_rmse']:+,.2f} | {row['manual_rmse_improvement_pct']:+.2f} | {row['auto_rmse_improvement_pct']:+.2f} | {row['auto_minus_manual_rmse_improvement_pct']:+.2f} |"
        )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_run_level_deltas(summary_df: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary_df.sort_values("auto_minus_manual_rmse_improvement_pct", ascending=True).copy()
    labels = [f"{row.dataset} | {row.model_label}" for row in ordered.itertuples()]
    colors = [COLORS["positive"] if value >= 0 else COLORS["negative"] for value in ordered["auto_minus_manual_rmse_improvement_pct"]]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(labels, ordered["auto_minus_manual_rmse_improvement_pct"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Auto minus manual RMSE improvement (percentage points)")
    ax.set_title("Categorical Mode Delta by Dataset and Model")
    fig.tight_layout()
    fig.savefig(output_dir / "auto_minus_manual_improvement_by_run.png")
    plt.close(fig)


def _plot_dataset_means(summary_df: pd.DataFrame, output_dir: Path) -> None:
    dataset_avg = (
        summary_df.groupby("dataset", as_index=False)
        .agg(mean_delta=("auto_minus_manual_rmse_improvement_pct", "mean"))
        .sort_values("mean_delta", ascending=False)
    )
    colors = [COLORS["positive"] if value >= 0 else COLORS["negative"] for value in dataset_avg["mean_delta"]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(dataset_avg["dataset"], dataset_avg["mean_delta"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean auto-minus-manual improvement (pp)")
    ax.set_title("Average Categorical Mode Delta by Dataset")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "auto_minus_manual_improvement_by_dataset.png")
    plt.close(fig)


def _plot_model_means(summary_df: pd.DataFrame, output_dir: Path) -> None:
    model_avg = (
        summary_df.groupby("model_label", as_index=False)
        .agg(mean_delta=("auto_minus_manual_rmse_improvement_pct", "mean"))
        .sort_values("mean_delta", ascending=False)
    )
    colors = [COLORS["positive"] if value >= 0 else COLORS["negative"] for value in model_avg["mean_delta"]]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(model_avg["model_label"], model_avg["mean_delta"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean auto-minus-manual improvement (pp)")
    ax.set_title("Average Categorical Mode Delta by Model")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(output_dir / "auto_minus_manual_improvement_by_model.png")
    plt.close(fig)


def _plot_winner_counts(summary_df: pd.DataFrame, output_dir: Path) -> None:
    counts = summary_df["winner"].value_counts().reindex(["auto", "manual", "tie"], fill_value=0)
    colors = [COLORS["auto"], COLORS["manual"], "#7f7f7f"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=colors)
    ax.set_ylabel("Run count")
    ax.set_title("Categorical Mode Winner Counts")
    fig.tight_layout()
    fig.savefig(output_dir / "winner_counts.png")
    plt.close(fig)


def generate_summary(run_dirs: list[Path], output_dir: Path) -> Path:
    rows = []
    for run_dir in run_dirs:
        comparison_df = _load_comparison(run_dir)
        rows.append(_build_row(comparison_df))

    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["dataset", "model_label"]).reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_dir / "categorical_mode_matrix.csv", index=False)
    _write_markdown(summary_df, output_dir)
    _plot_run_level_deltas(summary_df, output_dir)
    _plot_dataset_means(summary_df, output_dir)
    _plot_model_means(summary_df, output_dir)
    _plot_winner_counts(summary_df, output_dir)
    return output_dir


def _discover_latest_runs() -> list[Path]:
    run_dirs = sorted(RESULTS_DIR.glob("*_categorical_compare_*"))
    latest_by_key: dict[tuple[str, str], Path] = {}

    for run_dir in run_dirs:
        csv_path = run_dir / "data" / "categorical_mode_comparison.csv"
        if not csv_path.exists():
            continue
        comparison_df = pd.read_csv(csv_path)
        dataset = str(comparison_df["dataset"].iloc[0])
        model_type = str(comparison_df["model_type"].iloc[0])
        key = (dataset, model_type)
        if key not in latest_by_key or _run_timestamp(run_dir.name) > _run_timestamp(latest_by_key[key].name):
            latest_by_key[key] = run_dir

    return [latest_by_key[key] for key in sorted(latest_by_key)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize manual-vs-auto categorical comparison runs")
    parser.add_argument("run_dirs", nargs="*", type=Path, help="Specific categorical comparison directories")
    parser.add_argument("--latest", action="store_true", help="Use the latest run per dataset/model from results/")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory where summary artifacts will be written")
    args = parser.parse_args()

    run_dirs = args.run_dirs
    if args.latest or not run_dirs:
        run_dirs = _discover_latest_runs()
    if not run_dirs:
        raise FileNotFoundError("No categorical comparison runs found")

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = RESULTS_DIR / f"categorical_mode_batch_summary_{timestamp}"

    generated_dir = generate_summary(run_dirs, output_dir)
    print(f"Summary written to: {generated_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())