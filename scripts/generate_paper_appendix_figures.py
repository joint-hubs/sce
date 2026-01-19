#!/usr/bin/env python3
"""
@module: scripts.generate_paper_appendix_figures
@depends: results/*/data/*.csv
@exports: Main paper + appendix figures
@paper_ref: Figures B1, B2, B4, Appendix A1-A6
@data_flow: per-run csv -> matplotlib figures -> PDF/PNG

Generates consolidated main-paper figures and per-dataset appendix figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

COLORS = {
    "baseline": "#888888",
    "sce": "#2ecc71",
    "base": "#3498db",
    "context": "#e74c3c",
    "accent": "#7B1FA2",
}

SCE_SUFFIXES = (
    "_mean",
    "_median",
    "_std",
    "_count",
    "_ratio_to_general",
    "_p25",
    "_p75",
    "_p90",
    "_p10",
    "_min",
    "_max",
    "_range",
)


def is_sce_feature(feature_name: str) -> bool:
    return any(feature_name.endswith(suffix) for suffix in SCE_SUFFIXES)


def format_dataset(name: str) -> str:
    return name.replace("_", " ").title()


def fmt_num(val: float, decimals: int = 2) -> str:
    """Format number with commas and specified decimal places."""
    if pd.isna(val):
        return "—"
    return f"{val:,.{decimals}f}"


def save_figure(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(output_base.with_suffix(f".{ext}"))
    plt.close(fig)


def discover_run_dirs(results_root: Path) -> list[Path]:
    return sorted([
        p for p in results_root.iterdir()
        if p.is_dir() and "_search_" in p.name and (p / "data").exists()
    ])


def load_run_artifacts(run_dir: Path) -> dict[str, Any]:
    data_dir = run_dir / "data"
    dataset = run_dir.name.split("_search_")[0]
    artifacts = {
        "dataset": dataset,
        "run_dir": run_dir,
        "model_comparison": pd.read_csv(data_dir / "model_comparison.csv"),
        "best_by_strategy": pd.read_csv(data_dir / "best_by_strategy.csv"),
        "feature_importance": pd.read_csv(data_dir / "aggregated_feature_importance.csv"),
    }
    pruning_path = data_dir / "xgb_pruning_trace.csv"
    if pruning_path.exists():
        artifacts["pruning"] = pd.read_csv(pruning_path)
    stats_path = data_dir / "lm_context_statistics.csv"
    if stats_path.exists():
        artifacts["context_stats"] = pd.read_csv(stats_path)
    return artifacts


def best_baseline_and_sce(model_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    baseline = model_df[model_df["n_context"] == 0].sort_values("rmse").iloc[0]
    sce = model_df[(model_df["n_context"] > 0) & (model_df["n_base"] > 0)].sort_values("rmse").iloc[0]
    return baseline, sce


def plot_m1_rmse_improvement(runs: list[dict[str, Any]], output_base: Path) -> None:
    labels = [format_dataset(r["dataset"]) for r in runs]
    baseline_rmses = []
    sce_rmses = []
    improvements = []
    for r in runs:
        baseline, sce = best_baseline_and_sce(r["model_comparison"])
        baseline_rmses.append(baseline["rmse"])
        sce_rmses.append(sce["rmse"])
        improvements.append((baseline["rmse"] - sce["rmse"]) / baseline["rmse"] * 100)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width / 2, baseline_rmses, width, color=COLORS["baseline"],
                   edgecolor="black", linewidth=0.5, label="Baseline")
    bars2 = ax.bar(x + width / 2, sce_rmses, width, color=COLORS["sce"],
                   edgecolor="black", linewidth=0.5, label="Best SCE")
    ax.set_ylabel("RMSE (log scale)")
    ax.set_yscale("log")
    ax.set_title("RMSE: Baseline vs Best SCE", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(loc="upper left", fontsize=8)
    # Annotate bars with formatted values
    for bar, val in zip(bars1, baseline_rmses):
        ax.annotate(fmt_num(val, 0), xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    for bar, val in zip(bars2, sce_rmses):
        ax.annotate(fmt_num(val, 0), xy=(bar.get_x() + bar.get_width() / 2, val),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    save_figure(fig, output_base)

    # Create formatted table
    table_df = pd.DataFrame({
        "Dataset": labels,
        "Baseline RMSE": [fmt_num(v) for v in baseline_rmses],
        "Best SCE RMSE": [fmt_num(v) for v in sce_rmses],
        "Improvement %": [fmt_num(v) for v in improvements],
    })
    table_df.to_csv(output_base.with_name(f"{output_base.name}_table.csv"), index=False)
    save_table_figure(table_df, output_base.with_name(f"{output_base.name}_table"),
                      title="RMSE Summary Table")


def plot_m2_feature_contribution(runs: list[dict[str, Any]], output_base: Path) -> None:
    labels = [format_dataset(r["dataset"]) for r in runs]
    base_shares = []
    sce_shares = []
    for r in runs:
        df = r["feature_importance"]
        df = df.copy()
        df["is_sce"] = df["feature"].apply(is_sce_feature)
        total = df["importance_mean"].sum()
        if total == 0:
            base_shares.append(0)
            sce_shares.append(0)
        else:
            base_shares.append(df.loc[~df["is_sce"], "importance_mean"].sum() / total)
            sce_shares.append(df.loc[df["is_sce"], "importance_mean"].sum() / total)
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(labels))
    ax.bar(x, base_shares, label="Base", color=COLORS["base"], edgecolor="black", linewidth=0.5)
    ax.bar(x, sce_shares, bottom=base_shares, label="Context", color=COLORS["context"], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Share of Importance")
    ax.set_title("Feature Contribution Split", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_base)

    # Formatted table with percentages
    table_df = pd.DataFrame({
        "Dataset": labels,
        "Base Share": [f"{v*100:.2f}%" for v in base_shares],
        "Context Share": [f"{v*100:.2f}%" for v in sce_shares],
    })
    table_df.to_csv(output_base.with_name(f"{output_base.name}_table.csv"), index=False)
    save_table_figure(table_df, output_base.with_name(f"{output_base.name}_table"),
                      title="Feature Contribution Split")


def plot_m3_strategy_ranking(runs: list[dict[str, Any]], output_base: Path) -> None:
    datasets = [format_dataset(r["dataset"]) for r in runs]
    rank_frames = []
    for r in runs:
        df = r["best_by_strategy"].sort_values("rmse").reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        df["dataset"] = format_dataset(r["dataset"])
        rank_frames.append(df[["strategy", "rank", "dataset"]])
    all_ranks = pd.concat(rank_frames, ignore_index=True)
    avg_rank = all_ranks.groupby("strategy")["rank"].mean().sort_values()
    top_strategies = avg_rank.head(10).index.tolist()

    table = all_ranks[all_ranks["strategy"].isin(top_strategies)].copy()
    pivot = table.pivot(index="strategy", columns="dataset", values="rank").reindex(top_strategies)
    pivot["Avg"] = avg_rank.loc[top_strategies].round(1)

    # Save CSV
    csv_df = pivot.copy()
    csv_df.index.name = "Strategy"
    csv_df.to_csv(output_base.with_name(f"{output_base.name}_table.csv"))

    # Create heatmap figure
    fig, ax = plt.subplots(figsize=(8, 5))
    heatmap_data = pivot.values.astype(float)
    im = ax.imshow(heatmap_data, cmap="RdYlGn_r", aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))

    # Shorten dataset labels for x-axis
    short_labels = [c.replace("Rental ", "").replace("Sales ", "") if c != "Avg" else c for c in pivot.columns]
    ax.set_xticklabels(short_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 6 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=text_color, fontsize=9)

    ax.set_title("Strategy Ranking Heatmap (Lower is Better)", fontweight="bold", pad=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Rank")
    fig.tight_layout()
    save_figure(fig, output_base)


def save_table_figure(df: pd.DataFrame, output_base: Path, title: str) -> None:
    n_cols = len(df.columns)
    n_rows = len(df)
    fig_width = max(6, 1.0 * n_cols)
    fig_height = max(2, 0.4 * n_rows + 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(n_cols)))
    table.scale(1, 1.4)
    ax.set_title(title, fontweight="bold", pad=15)
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_a1_baseline_vs_sce(run: dict[str, Any], output_base: Path) -> None:
    baseline, sce = best_baseline_and_sce(run["model_comparison"])
    fig, ax = plt.subplots(figsize=(4.5, 3))
    labels = ["Baseline", "Best SCE"]
    values = [baseline["rmse"], sce["rmse"]]
    bars = ax.bar(labels, values, color=[COLORS["baseline"], COLORS["sce"]], edgecolor="black", linewidth=0.5)
    ax.set_ylabel("RMSE")
    ax.set_title(f"{format_dataset(run['dataset'])}: Baseline vs SCE", fontweight="bold")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,.0f}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, output_base)


def plot_a2_strategy_ladder(run: dict[str, Any], output_base: Path) -> None:
    df = run["best_by_strategy"].sort_values("rmse")
    fig, ax = plt.subplots(figsize=(6, 4))
    y = np.arange(len(df))
    ax.barh(y, df["rmse"], color=COLORS["sce"], edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(df["strategy"], fontsize=8)
    ax.set_xlabel("RMSE")
    ax.set_title(f"{format_dataset(run['dataset'])}: Strategy RMSE Ladder", fontweight="bold")
    ax.invert_yaxis()
    save_figure(fig, output_base)


def plot_a3_top_features(run: dict[str, Any], output_base: Path, top_n: int = 20) -> None:
    df = run["feature_importance"].sort_values("importance_mean", ascending=False).head(top_n)
    df = df.copy()
    df["is_sce"] = df["feature"].apply(is_sce_feature)
    colors = [COLORS["context"] if v else COLORS["base"] for v in df["is_sce"]]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    y = np.arange(len(df))
    ax.barh(y, df["importance_mean"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"], fontsize=7)
    ax.set_xlabel("Mean Importance")
    ax.set_title(f"{format_dataset(run['dataset'])}: Top {top_n} Features", fontweight="bold")
    ax.invert_yaxis()
    legend_handles = [
        plt.Line2D([0], [0], color=COLORS["base"], lw=6, label="Base features"),
        plt.Line2D([0], [0], color=COLORS["context"], lw=6, label="Context (SCE) features"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8)
    save_figure(fig, output_base)


def plot_a4_ablation_curve(run: dict[str, Any], output_base: Path) -> None:
    if "pruning" not in run:
        return
    df = run["pruning"].sort_values("n_features")
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(df["n_features"], df["rmse"], marker="o", color=COLORS["accent"], linewidth=1.5)
    ax.set_xlabel("# Features")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{format_dataset(run['dataset'])}: Ablation Curve", fontweight="bold")
    save_figure(fig, output_base)


def plot_a5_context_tstats(run: dict[str, Any], output_base: Path) -> None:
    if "context_stats" not in run:
        return
    df = run["context_stats"]
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.hist(df["t_statistic"], bins=30, color=COLORS["context"], edgecolor="white", alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("t-statistic")
    ax.set_ylabel("Count")
    ax.set_title(f"{format_dataset(run['dataset'])}: Context t-Stats", fontweight="bold")
    save_figure(fig, output_base)


def plot_a6_complexity_vs_rmse(run: dict[str, Any], output_base: Path) -> None:
    df = run["model_comparison"].copy()
    df["is_sce"] = df["n_context"] > 0
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.scatter(df.loc[~df["is_sce"], "n_features"], df.loc[~df["is_sce"], "rmse"],
               color=COLORS["baseline"], alpha=0.6, label="Baseline")
    ax.scatter(df.loc[df["is_sce"], "n_features"], df.loc[df["is_sce"], "rmse"],
               color=COLORS["sce"], alpha=0.6, label="SCE")
    ax.set_xlabel("# Features")
    ax.set_ylabel("RMSE")
    ax.set_title(f"{format_dataset(run['dataset'])}: Complexity vs RMSE", fontweight="bold")
    ax.legend(loc="upper right", fontsize=8)
    save_figure(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate main paper + appendix figures")
    parser.add_argument("--results-root", type=Path, default=Path(__file__).parent.parent / "results")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent.parent.parent / "docs" / "figures")
    parser.add_argument("--paper-only", action="store_true", help="Only generate main paper figures")
    parser.add_argument("--appendix-only", action="store_true", help="Only generate appendix figures")
    parser.add_argument("--dataset", type=str, default=None, help="Limit to a specific dataset")
    args = parser.parse_args()

    run_dirs = discover_run_dirs(args.results_root)
    if args.dataset:
        run_dirs = [d for d in run_dirs if d.name.startswith(args.dataset)]

    runs = [load_run_artifacts(d) for d in run_dirs]
    if not runs:
        raise SystemExit("No run directories found for figure generation.")

    paper_dir = args.output_root / "paper"
    appendix_dir = args.output_root / "appendix"

    if not args.appendix_only:
        plot_m1_rmse_improvement(runs, paper_dir / "paper_m1_rmse_improvement")
        plot_m2_feature_contribution(runs, paper_dir / "paper_m2_feature_contributions")
        plot_m3_strategy_ranking(runs, paper_dir / "paper_m3_strategy_ranking")

    if not args.paper_only:
        for run in runs:
            dataset = run["dataset"]
            base = appendix_dir / f"appendix_{dataset}_A"
            plot_a1_baseline_vs_sce(run, base.with_name(f"appendix_{dataset}_A1"))
            plot_a2_strategy_ladder(run, base.with_name(f"appendix_{dataset}_A2"))
            plot_a3_top_features(run, base.with_name(f"appendix_{dataset}_A3"))
            plot_a4_ablation_curve(run, base.with_name(f"appendix_{dataset}_A4"))
            plot_a5_context_tstats(run, base.with_name(f"appendix_{dataset}_A5"))
            plot_a6_complexity_vs_rmse(run, base.with_name(f"appendix_{dataset}_A6"))


if __name__ == "__main__":
    main()
