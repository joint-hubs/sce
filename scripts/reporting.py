"""
@module: scripts.reporting
@depends: pandas, matplotlib, seaborn
@exports: generate_search_reports
@data_flow: model_comparison.csv -> figures + tables
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import json

import pandas as pd


def _write_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, separator] + body_lines)


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def _load_metadata(output_dir: Path) -> dict:
    metadata_path = output_dir / "data" / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {}


def _write_run_readme(
    output_dir: Path,
    strategy_table: pd.DataFrame,
    top_strategies: list[dict],
) -> None:
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    metadata = _load_metadata(output_dir)
    dataset = metadata.get("dataset", output_dir.name.split("_search_")[0])

    md_lines = [
        f"# Run Report: {dataset}",
        "",
        f"Run directory: {output_dir.name}",
        "",
        "## Strategy Accuracy (best per strategy)",
        "",
    ]

    rows = []
    for _, row in strategy_table.iterrows():
        rows.append([
            str(int(row["rank"])),
            str(row["strategy"]),
            f"{row['rmse']:.2f}",
            f"{row['r2']:.4f}",
            f"{row['mae']:.2f}",
            str(int(row["n_features"])),
            str(row["model_config"]),
        ])
    md_lines.append(_write_markdown_table(
        ["Rank", "Strategy", "RMSE", "R2", "MAE", "Features", "Model"],
        rows,
    ))

    md_lines.extend([
        "",
        "## Top 3 Strategies: Features + Impact",
        "",
    ])

    for entry in top_strategies:
        md_lines.append(f"### {entry['strategy']} (Rank {entry['rank']})")
        md_lines.append("")
        md_lines.append(f"- Features list: {entry['features_csv']}")
        md_lines.append(f"- Feature impact: {entry['impact_csv']}")
        if entry.get("impact_md"):
            md_lines.append(f"- Feature impact summary: {entry['impact_md']}")
        md_lines.append("")

    if figures_dir.exists():
        figures = sorted(p.name for p in figures_dir.glob("*.png"))
        if figures:
            md_lines.extend(["## Visualizations", ""])
            for fig in figures:
                md_lines.append(f"- figures/{fig}")

    if data_dir.exists():
        md_lines.extend(["", "## Data Artifacts", ""])
        artifacts = sorted(p.name for p in data_dir.glob("*.csv"))
        for artifact in artifacts:
            md_lines.append(f"- data/{artifact}")

    (output_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")


def _update_root_readme(output_dir: Path) -> None:
    root_dir = output_dir.parent.parent
    readme_path = root_dir / "README.md"
    if not readme_path.exists():
        return

    start_marker = "<!-- SCE_RUN_REPORTS:START -->"
    end_marker = "<!-- SCE_RUN_REPORTS:END -->"

    content = readme_path.read_text(encoding="utf-8")
    if start_marker not in content or end_marker not in content:
        return

    results_dir = root_dir / "results"
    report_paths = sorted(
        (p for p in results_dir.glob("*_search_*") if (p / "README.md").exists()),
        key=lambda p: p.name,
        reverse=True,
    )
    report_lines = []
    for path in report_paths[:10]:
        rel = path / "README.md"
        rel_path = rel.relative_to(root_dir).as_posix()
        report_lines.append(f"- [{path.name}]({rel_path})")

    block = "\n".join([start_marker] + report_lines + [end_marker])
    before = content.split(start_marker)[0]
    after = content.split(end_marker)[1]
    readme_path.write_text(before + block + after, encoding="utf-8")


def generate_search_reports(results_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate reporting figures and tables for search results."""
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save top-5 configs table
    top5 = results_df.sort_values("rmse").head(5)
    top5.to_csv(data_dir / "top_5_configs.csv", index=False)

    # Best per strategy
    if "strategy" in results_df.columns:
        best_by_strategy = results_df.sort_values("rmse").groupby("strategy").head(1)
        best_by_strategy.to_csv(data_dir / "best_by_strategy.csv", index=False)

        # Strategy comparison table (ranked by RMSE)
        strategy_table = best_by_strategy.sort_values("rmse").copy()
        strategy_table.insert(0, "rank", range(1, len(strategy_table) + 1))
        strategy_table = strategy_table[["rank", "strategy", "rmse", "r2", "mae", "n_features", "model_config", "features"]]
        strategy_table.to_csv(data_dir / "strategy_comparison.csv", index=False)

        # Markdown version
        md_lines = ["# Strategy Comparison", ""]
        rows = []
        for _, row in strategy_table.iterrows():
            rows.append([
                str(int(row["rank"])),
                str(row["strategy"]),
                f"{row['rmse']:.2f}",
                f"{row['r2']:.4f}",
                f"{row['mae']:.2f}",
                str(int(row["n_features"])),
                str(row["model_config"]),
            ])
        md_lines.append(_write_markdown_table(
            ["Rank", "Strategy", "RMSE", "R2", "MAE", "Features", "Model"],
            rows,
        ))
        md_lines.append("")
        md_lines.append("Note: R2 uses ASCII notation for compatibility on Windows.")
        (data_dir / "strategy_comparison.md").write_text("\n".join(md_lines), encoding="utf-8")

        # Top-3 strategies: feature attachments + impact
        agg_path = data_dir / "aggregated_feature_importance.csv"
        agg_importance = pd.read_csv(agg_path) if agg_path.exists() else pd.DataFrame()
        top_strategies: list[dict] = []
        for _, row in strategy_table.head(3).iterrows():
            strategy = str(row["strategy"])
            rank = int(row["rank"])
            features = [f for f in str(row["features"]).split("|") if f]
            slug = _safe_slug(strategy)

            features_df = pd.DataFrame({"feature": features})
            features_csv_name = f"top_{rank}_{slug}_features.csv"
            features_df.to_csv(data_dir / features_csv_name, index=False)

            impact_csv_name = f"top_{rank}_{slug}_feature_impact.csv"
            impact_md_name = f"top_{rank}_{slug}_feature_impact.md"
            impact_df = pd.DataFrame()

            if not agg_importance.empty:
                if strategy in agg_importance.columns:
                    impact_col = strategy
                else:
                    impact_col = "importance_mean"

                impact_df = agg_importance[
                    agg_importance["feature"].isin(features)
                ][["feature", impact_col, "importance_std", "n_models"]].rename(
                    columns={impact_col: "importance"}
                )
                impact_df = impact_df.sort_values("importance", ascending=False)
                total = float(impact_df["importance"].sum()) if not impact_df.empty else 0.0
                impact_df["impact_share"] = impact_df["importance"].apply(
                    lambda v: float(v) / total if total else 0.0
                )
                impact_df.to_csv(data_dir / impact_csv_name, index=False)

                impact_rows = []
                for _, imp in impact_df.head(20).iterrows():
                    impact_rows.append([
                        str(imp["feature"]),
                        f"{imp['importance']:.6f}",
                        f"{imp['impact_share']:.4f}",
                        str(int(imp["n_models"])),
                    ])
                impact_md = [
                    f"# Feature Impact: {strategy}",
                    "",
                    _write_markdown_table(
                        ["Feature", "Importance", "Impact Share", "Models"],
                        impact_rows,
                    ),
                ]
                (data_dir / impact_md_name).write_text("\n".join(impact_md), encoding="utf-8")
            else:
                pd.DataFrame({"feature": features, "importance": [0.0] * len(features)}).to_csv(
                    data_dir / impact_csv_name,
                    index=False,
                )

            top_strategies.append({
                "strategy": strategy,
                "rank": rank,
                "features_csv": f"data/{features_csv_name}",
                "impact_csv": f"data/{impact_csv_name}",
                "impact_md": f"data/{impact_md_name}" if not impact_df.empty else "",
            })

        _write_run_readme(output_dir, strategy_table, top_strategies)
        _update_root_readme(output_dir)

    # Optional plotting
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception:
        return

    # RMSE by strategy
    if "strategy" in results_df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=results_df, x="strategy", y="rmse", palette="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.title("RMSE Distribution by Strategy")
        plt.tight_layout()
        plt.savefig(figures_dir / "comparison_strategy_rmse.png", dpi=150)
        plt.close()

    # Performance vs number of features
    if "n_features" in results_df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=results_df, x="n_features", y="rmse", hue="strategy", alpha=0.6)
        plt.title("Performance vs Model Complexity")
        plt.xlabel("Number of Features")
        plt.ylabel("RMSE")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(figures_dir / "comparison_complexity_vs_rmse.png", dpi=150)
        plt.close()

    # Strategy matrix heatmap (subset experiments)
    if "strategy" in results_df.columns:
        subset_results = results_df[results_df["strategy"].str.contains("filtered_", na=False)].copy()
        if not subset_results.empty:
            subset_results["subset_name"] = subset_results["strategy"].apply(lambda x: x.split("_")[-1])
            subset_results["base_strategy"] = subset_results["strategy"].apply(lambda x: x.split("_")[0])
            pivot_rmse = subset_results.pivot_table(
                index="base_strategy",
                columns="subset_name",
                values="rmse",
                aggfunc="min",
            )
            plt.figure(figsize=(10, 6))
            sns.heatmap(pivot_rmse, annot=True, fmt=",.0f", cmap="YlGnBu")
            plt.title("Strategy Matrix: Best RMSE by Subset and Filter")
            plt.tight_layout()
            plt.savefig(figures_dir / "comparison_strategy_matrix_heatmap.png", dpi=150)
            plt.close()
