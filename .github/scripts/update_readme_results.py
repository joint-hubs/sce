#!/usr/bin/env python3
"""
Update results README with experiment results and figures.

This script is called by GitHub Actions after experiments complete.
It injects:
    - Results summary table
    - Performance metrics
    - Links to generated figures

Usage:
        python update_readme_results.py \
                --results artifacts/experiment_results.json \
                --figures artifacts/figures/ \
                --readme docs/results/README.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime


def load_results(results_path: Path) -> list[dict]:
    """Load experiment results from JSON."""
    with open(results_path) as f:
        return json.load(f)


def format_results_table(results: list[dict]) -> str:
    """Generate markdown table from results."""
    lines = [
        "| Dataset | Samples | RMSE Δ | Baseline R² | SCE R² | R² Δ |",
        "|---------|--------:|-------:|------------:|-------:|-----:|",
    ]
    
    for r in results:
        name = r['dataset'].replace('_', ' ').title()
        samples = f"{r['n_samples']:,}"
        baseline_r2 = f"{r['baseline_r2']:.4f}"
        sce_r2 = f"{r['sce_r2']:.4f}"
        rmse_delta = f"**{r['rmse_improvement_pct']:+.1f}%**"
        r2_delta = f"{r['r2_improvement_pct']:+.1f}%"
        
        lines.append(f"| {name} | {samples} | {rmse_delta} | {baseline_r2} | {sce_r2} | {r2_delta} |")
    
    # Average row
    avg_rmse = sum(r['rmse_improvement_pct'] for r in results) / len(results)
    avg_r2 = sum(r['r2_improvement_pct'] for r in results) / len(results)
    lines.append(f"| **Average** | — | **{avg_rmse:+.1f}%** | — | — | **{avg_r2:+.1f}%** |")
    
    return "\n".join(lines)


def format_summary_badges(results: list[dict], links: dict[str, str]) -> str:
    """Generate summary statistics as badges."""
    avg_rmse = sum(r['rmse_improvement_pct'] for r in results) / len(results)
    total_samples = sum(r['n_samples'] for r in results)
    
    # Format for shields.io badges
    rmse_color = "brightgreen" if avg_rmse > 5 else "green" if avg_rmse > 2 else "yellow"
    
    return f"""[![RMSE Improvement](https://img.shields.io/badge/Avg_RMSE_Improvement-{avg_rmse:.1f}%25-{rmse_color})]({links['results_page']})
[![Datasets](https://img.shields.io/badge/Datasets_Tested-{len(results)}-blue)]({links['workflow']})
[![Samples](https://img.shields.io/badge/Total_Samples-{total_samples:,}-informational)]({links['data']})"""


def rel_link(target: Path, readme_path: Path) -> str:
    target_path = (Path.cwd() / target).resolve()
    readme_dir = readme_path.resolve().parent
    relative = os.path.relpath(target_path, readme_dir)
    return relative.replace("\\", "/")


def update_readme_section(
    readme_content: str,
    section_marker: str,
    new_content: str
) -> str:
    """
    Update a section in README marked by HTML comments.
    
    Sections are marked like:
    <!-- RESULTS_START -->
    ... content to replace ...
    <!-- RESULTS_END -->
    """
    start_marker = f"<!-- {section_marker}_START -->"
    end_marker = f"<!-- {section_marker}_END -->"
    
    pattern = re.compile(
        rf"({re.escape(start_marker)}).*?({re.escape(end_marker)})",
        re.DOTALL
    )
    
    replacement = f"{start_marker}\n{new_content}\n{end_marker}"
    
    if pattern.search(readme_content):
        return pattern.sub(replacement, readme_content)
    else:
        # Markers don't exist - append to file (or log warning)
        print(f"Warning: Section markers {section_marker} not found in README")
        return readme_content


def main():
    parser = argparse.ArgumentParser(description="Update README with experiment results")
    parser.add_argument("--results", "-r", type=Path, required=True,
                        help="Path to experiment_results.json")
    parser.add_argument("--figures", "-f", type=Path,
                        help="Path to figures directory")
    parser.add_argument("--readme", "-m", type=Path, default=Path("docs/results/README.md"),
                        help="Path to results README")
    
    args = parser.parse_args()
    
    # Load results
    if not args.results.exists():
        print(f"Error: Results file not found: {args.results}")
        return 1
    
    results = load_results(args.results)
    print(f"Loaded {len(results)} experiment results")
    
    # Read current README
    readme_content = args.readme.read_text(encoding="utf-8")
    
    # Generate new content
    results_table = format_results_table(results)
    links = {
        "results_page": rel_link(Path("docs/results/README.md"), args.readme),
        "workflow": rel_link(Path(".github/workflows/run-experiments.yml"), args.readme),
        "data": rel_link(Path("data"), args.readme),
        "figures": rel_link(Path("docs/figures/results"), args.readme),
    }
    summary_badges = format_summary_badges(results, links)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    
    # Update sections
    readme_content = update_readme_section(
        readme_content, 
        "RESULTS_TABLE",
        f"{results_table}\n\n*Last updated: {timestamp}*"
    )
    
    readme_content = update_readme_section(
        readme_content,
        "RESULTS_BADGES", 
        summary_badges
    )
    
    # Figure references
    if args.figures and args.figures.exists():
        figures = list(args.figures.glob("*.png"))
        if figures:
            figure_section = "\n### 📊 Latest Results\n\n"
            for fig in sorted(figures)[:3]:  # Top 3 figures
                fig_name = fig.stem.replace("_", " ").title()
                figure_section += f"![{fig_name}]({links['figures']}/{fig.name})\n\n"
            
            readme_content = update_readme_section(
                readme_content,
                "RESULTS_FIGURES",
                figure_section
            )
    
    # Write updated README
    args.readme.write_text(readme_content, encoding="utf-8")
    print(f"Updated {args.readme}")
    
    return 0


if __name__ == "__main__":
    exit(main())
