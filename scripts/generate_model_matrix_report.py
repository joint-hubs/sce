#!/usr/bin/env python3
"""
@module: scripts.generate_model_matrix_report
@depends: pandas, matplotlib
@exports: generate_model_matrix_report
@data_flow: model_matrix csv -> per-dataset summaries and plots

Generate per-dataset summary tables and plots from a full model sweep.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
    "baseline": "#4A7BA7",
    "sce": "#5DA271",
    "positive": "#E07B39",
    "negative": "#B24C63",
}

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "model_matrix_20260329_235549"

GROUP_LABELS = {
    "model_type": {
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "catboost": "CatBoost",
        "ridge": "Ridge",
        "random_forest": "Random Forest",
        "extra_trees": "Extra Trees",
        "gradient_boosting": "Gradient Boosting",
    },
    "context_variant": {
        "sce": "SCE",
        "target_mean": "Target Mean",
        "hierarchical_mean_count": "Hierarchical Mean+Count",
        "hierarchical_mean_std_count": "Hierarchical Mean+Std+Count",
    },
}


def _format_dataset_name(dataset: str) -> str:
    return dataset.replace("_", " ").title()


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def _format_group_value(group_column: str, value: str) -> str:
    return GROUP_LABELS.get(group_column, {}).get(value, value.replace("_", " ").title())


def _load_results(results_dir: Path) -> pd.DataFrame:
    csv_path = results_dir / "all_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find results file: {csv_path}")

    df = pd.read_csv(csv_path, decimal=",")

    alias_map = {
        "enriched_rmse": "sce_rmse",
        "enriched_r2": "sce_r2",
        "n_enriched_features": "n_sce_features",
    }
    for source_col, target_col in alias_map.items():
        if source_col in df.columns and target_col not in df.columns:
            df[target_col] = df[source_col]

    numeric_cols = [
        "baseline_rmse",
        "baseline_r2",
        "sce_rmse",
        "sce_r2",
        "rmse_improvement_pct",
        "r2_improvement_pct",
        "n_samples",
        "n_baseline_features",
        "n_sce_features",
        "runtime_seconds",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    sort_col = "model_type" if "model_type" in df.columns else "context_variant"
    return df.sort_values(["dataset", sort_col]).reset_index(drop=True)


def _build_dataset_markdown(dataset_df: pd.DataFrame, group_column: str) -> str:
    dataset = dataset_df["dataset"].iloc[0]
    title = _format_dataset_name(dataset)
    best_rmse = dataset_df.loc[dataset_df["sce_rmse"].idxmin()]
    best_delta = dataset_df.loc[dataset_df["rmse_improvement_pct"].idxmax()]
    group_title = "Model" if group_column == "model_type" else "Context Variant"

    lines = [
        f"# {title}",
        "",
        f"- Best absolute SCE RMSE: `{_format_group_value(group_column, best_rmse[group_column])}` at `{best_rmse['sce_rmse']:,.2f}`",
        f"- Best RMSE improvement: `{_format_group_value(group_column, best_delta[group_column])}` at `{best_delta['rmse_improvement_pct']:+.2f}%`",
        "",
        f"| {group_title} | Baseline RMSE | SCE RMSE | RMSE Improvement % | Baseline R2 | SCE R2 | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    ordered = dataset_df.sort_values("sce_rmse")
    for _, row in ordered.iterrows():
        lines.append(
            "| {model} | {baseline_rmse:,.2f} | {sce_rmse:,.2f} | {improvement:+.2f} | {baseline_r2:.4f} | {sce_r2:.4f} | {runtime:.2f} |".format(
                model=_format_group_value(group_column, row[group_column]),
                baseline_rmse=row["baseline_rmse"],
                sce_rmse=row["sce_rmse"],
                improvement=row["rmse_improvement_pct"],
                baseline_r2=row["baseline_r2"],
                sce_r2=row["sce_r2"],
                runtime=row["runtime_seconds"],
            )
        )

    return "\n".join(lines) + "\n"


def _plot_dataset(dataset_df: pd.DataFrame, output_path: Path, group_column: str) -> None:
    ordered = dataset_df.sort_values("sce_rmse").reset_index(drop=True)
    groups = [_format_group_value(group_column, value) for value in ordered[group_column].tolist()]
    x = np.arange(len(groups))
    width = 0.38
    group_title = "Model" if group_column == "model_type" else "Context Variant"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1 = axes[0]
    ax1.bar(
        x - width / 2,
        ordered["baseline_rmse"],
        width,
        color=COLORS["baseline"],
        edgecolor="black",
        linewidth=0.5,
        label="Baseline",
    )
    ax1.bar(
        x + width / 2,
        ordered["sce_rmse"],
        width,
        color=COLORS["sce"],
        edgecolor="black",
        linewidth=0.5,
        label="+ SCE",
    )
    ax1.set_title(f"{_format_dataset_name(ordered['dataset'].iloc[0])}: RMSE by {group_title}", fontweight="bold")
    ax1.set_ylabel("RMSE")
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, rotation=25, ha="right")
    ax1.legend(loc="upper right")

    ax2 = axes[1]
    improvement_colors = [COLORS["positive"] if value >= 0 else COLORS["negative"] for value in ordered["rmse_improvement_pct"]]
    bars = ax2.bar(
        x,
        ordered["rmse_improvement_pct"],
        color=improvement_colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title(f"{_format_dataset_name(ordered['dataset'].iloc[0])}: RMSE Improvement by {group_title}", fontweight="bold")
    ax2.set_ylabel("Improvement (%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, rotation=25, ha="right")

    for bar, value in zip(bars, ordered["rmse_improvement_pct"]):
        offset = 0.2 if value >= 0 else -0.4
        va = "bottom" if value >= 0 else "top"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.1f}%",
            ha="center",
            va=va,
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"))
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _write_overview(df: pd.DataFrame, output_dir: Path, group_column: str) -> None:
    summary_rows = []
    for dataset, dataset_df in df.groupby("dataset", sort=True):
        best_rmse = dataset_df.loc[dataset_df["sce_rmse"].idxmin()]
        best_delta = dataset_df.loc[dataset_df["rmse_improvement_pct"].idxmax()]
        worst_delta = dataset_df.loc[dataset_df["rmse_improvement_pct"].idxmin()]
        summary_rows.append(
            {
                "dataset": dataset,
                "best_sce_rmse_group": _format_group_value(group_column, best_rmse[group_column]),
                "best_sce_rmse": best_rmse["sce_rmse"],
                "best_improvement_group": _format_group_value(group_column, best_delta[group_column]),
                "best_improvement_pct": best_delta["rmse_improvement_pct"],
                "worst_improvement_group": _format_group_value(group_column, worst_delta[group_column]),
                "worst_improvement_pct": worst_delta["rmse_improvement_pct"],
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("dataset")
    summary_df.to_csv(output_dir / "dataset_overview.csv", index=False)
    group_title = "Model" if group_column == "model_type" else "Context Variant"

    lines = [
        f"# Dataset {group_title} Overview",
        "",
        f"| Dataset | Best SCE RMSE {group_title} | Best SCE RMSE | Best Improvement {group_title} | Best Improvement % | Worst Improvement {group_title} | Worst Improvement % |",
        "|---|---|---:|---|---:|---|---:|",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            "| {dataset} | {best_sce_rmse_group} | {best_sce_rmse:,.2f} | {best_improvement_group} | {best_improvement_pct:+.2f} | {worst_improvement_group} | {worst_improvement_pct:+.2f} |".format(
                **row.to_dict()
            )
        )

    (output_dir / "dataset_overview.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_model_matrix_report(results_dir: Path, group_column: str = "model_type") -> Path:
    df = _load_results(results_dir)
    if group_column not in df.columns:
        raise ValueError(f"Column '{group_column}' not found in results. Available columns: {', '.join(df.columns)}")
    output_dir = results_dir / "dataset_reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_overview(df, output_dir, group_column)

    for dataset, dataset_df in df.groupby("dataset", sort=True):
        dataset_slug = _slugify(dataset)
        dataset_df.sort_values("sce_rmse").to_csv(output_dir / f"{dataset_slug}_models.csv", index=False)
        (output_dir / f"{dataset_slug}_summary.md").write_text(
            _build_dataset_markdown(dataset_df, group_column),
            encoding="utf-8",
        )
        _plot_dataset(dataset_df, output_dir / f"{dataset_slug}_comparison", group_column)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate per-dataset model sweep reports")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing all_results.csv from a model sweep",
    )
    parser.add_argument(
        "--group-column",
        choices=["model_type", "context_variant"],
        default="model_type",
        help="Column to compare within each dataset",
    )
    args = parser.parse_args()

    output_dir = generate_model_matrix_report(args.results_dir, group_column=args.group_column)
    print(f"Generated dataset reports in: {output_dir}")


if __name__ == "__main__":
    main()