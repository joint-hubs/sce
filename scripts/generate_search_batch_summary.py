"""
@module: scripts.generate_search_batch_summary
@depends: pandas, matplotlib
@exports: main
@data_flow: search run dirs -> combined summary tables + figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_run_summary(run_dir: Path) -> dict[str, object]:
    data_dir = run_dir / "data"
    best_by_strategy = pd.read_csv(data_dir / "best_by_strategy.csv")
    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    dataset = str(metadata.get("dataset") or run_dir.name.split("_search_")[0])
    ranked = best_by_strategy.sort_values("rmse").reset_index(drop=True)
    best_row = ranked.iloc[0]
    baseline_rows = ranked[ranked["strategy"] == "baseline"]
    baseline_row = baseline_rows.iloc[0] if not baseline_rows.empty else None

    baseline_rmse = float(baseline_row["rmse"]) if baseline_row is not None else float("nan")
    baseline_r2 = float(baseline_row["r2"]) if baseline_row is not None else float("nan")
    improvement_pct = (
        100.0 * (baseline_rmse - float(best_row["rmse"])) / baseline_rmse
        if pd.notna(baseline_rmse) and baseline_rmse != 0.0
        else float("nan")
    )

    return {
        "dataset": dataset,
        "run_dir": run_dir.name,
        "run_path": str(run_dir),
        "best_strategy": str(best_row["strategy"]),
        "best_model_config": str(best_row["model_config"]),
        "best_rmse": float(best_row["rmse"]),
        "best_r2": float(best_row["r2"]),
        "best_mae": float(best_row["mae"]),
        "best_n_features": int(best_row["n_features"]),
        "baseline_rmse": baseline_rmse,
        "baseline_r2": baseline_r2,
        "rmse_improvement_pct": improvement_pct,
        "best_by_strategy": ranked,
    }


def _write_summary_markdown(summary_df: pd.DataFrame, output_dir: Path) -> None:
    top = summary_df.sort_values("rmse_improvement_pct", ascending=False).reset_index(drop=True)
    win_counts = top["best_strategy"].value_counts().sort_values(ascending=False)
    mean_improvement = top["rmse_improvement_pct"].mean()

    lines = [
        "# Search Batch Summary",
        "",
        f"Runs summarized: {len(top)}",
        f"Average RMSE improvement over baseline: {mean_improvement:.2f}%",
        "",
        "## Strategy Wins",
        "",
    ]

    for strategy, wins in win_counts.items():
        lines.append(f"- {strategy}: {wins} datasets")

    lines.extend([
        "",
        "## Dataset Winners",
        "",
        "| Dataset | Best Strategy | Model | Best RMSE | Baseline RMSE | RMSE Improvement % | Best R2 | Features |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ])

    for _, row in top.iterrows():
        lines.append(
            "| {dataset} | {best_strategy} | {best_model_config} | {best_rmse:.2f} | {baseline_rmse:.2f} | {rmse_improvement_pct:.2f} | {best_r2:.4f} | {best_n_features} |".format(
                **row.to_dict()
            )
        )

    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_best_vs_baseline(summary_df: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary_df.sort_values("rmse_improvement_pct", ascending=False)
    x = range(len(ordered))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - 0.18 for i in x], ordered["baseline_rmse"], width=0.36, label="baseline", color="#d95f02")
    ax.bar([i + 0.18 for i in x], ordered["best_rmse"], width=0.36, label="best", color="#1b9e77")
    ax.set_xticks(list(x))
    ax.set_xticklabels(ordered["dataset"], rotation=25, ha="right")
    ax.set_ylabel("RMSE")
    ax.set_title("Baseline vs Best Search RMSE")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "baseline_vs_best_rmse.png", dpi=150)
    plt.close(fig)


def _plot_improvements(summary_df: pd.DataFrame, output_dir: Path) -> None:
    ordered = summary_df.sort_values("rmse_improvement_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(ordered["dataset"], ordered["rmse_improvement_pct"], color="#1f78b4")
    ax.set_xlabel("RMSE improvement over baseline (%)")
    ax.set_title("Search Winner Improvement by Dataset")
    fig.tight_layout()
    fig.savefig(output_dir / "rmse_improvement_by_dataset.png", dpi=150)
    plt.close(fig)


def _plot_strategy_wins(summary_df: pd.DataFrame, output_dir: Path) -> None:
    counts = summary_df["best_strategy"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(counts.index, counts.values, color="#7570b3")
    ax.set_ylabel("Datasets won")
    ax.set_title("Winning Strategy Counts")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_dir / "strategy_win_counts.png", dpi=150)
    plt.close(fig)


def generate_summary(run_dirs: list[Path], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    per_strategy_rows = []

    for run_dir in run_dirs:
        summary = _load_run_summary(run_dir)
        ranked = summary.pop("best_by_strategy")
        rows.append(summary)

        per_strategy = ranked.copy()
        per_strategy.insert(0, "dataset", summary["dataset"])
        per_strategy.insert(1, "run_dir", summary["run_dir"])
        per_strategy_rows.append(per_strategy)

    summary_df = pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)
    strategy_df = pd.concat(per_strategy_rows, ignore_index=True)
    win_counts = summary_df["best_strategy"].value_counts().rename_axis("best_strategy").reset_index(name="wins")

    summary_df.to_csv(output_dir / "dataset_best_summary.csv", index=False)
    strategy_df.to_csv(output_dir / "combined_best_by_strategy.csv", index=False)
    win_counts.to_csv(output_dir / "strategy_win_counts.csv", index=False)

    _write_summary_markdown(summary_df, output_dir)
    _plot_best_vs_baseline(summary_df, output_dir)
    _plot_improvements(summary_df, output_dir)
    _plot_strategy_wins(summary_df, output_dir)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize an explicit batch of search runs")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="Search result directories to summarize")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where combined tables, markdown, and figures will be written",
    )
    args = parser.parse_args()

    generate_summary(args.run_dirs, args.output_dir)
    print(f"Summary written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())