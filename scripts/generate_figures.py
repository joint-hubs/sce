#!/usr/bin/env python3
"""
@module: scripts.generate_figures
@depends: results/experiment_results.json
@exports: M1, M2, M3 figures
@paper_ref: Figures M1, M2, M3
@data_flow: experiment_results.json -> matplotlib figures -> PDF/PNG

Generates publication-quality figures for the SCE paper.
- B1: Consolidated performance bar charts (RMSE, R² improvements)
- B4: Feature importance analysis (base vs context features)
- B2: Hierarchy depth ablation study
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

# Publication style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette (accessible)
COLORS = {
    'baseline': '#4A7BA7',  # Blue
    'sce': '#5DA271',       # Green
    'improvement': '#E07B39' # Orange
}

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_FILE = PROJECT_ROOT / "results" / "experiment_results.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "figures"


def load_results() -> list[dict[str, Any]]:
    """Load experiment results from JSON file."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Results not found at {RESULTS_FILE}. "
            "Run 'python scripts/run.py --all' first."
        )
    with open(RESULTS_FILE) as f:
        return json.load(f)


def generate_b1_performance(results: list[dict], output_path: Path) -> None:
    """
    Generate B1: Consolidated Performance Bar Charts.
    
    Two-panel figure showing:
    - Left: RMSE comparison (baseline vs SCE)
    - Right: R² comparison (baseline vs SCE)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    datasets = [r['dataset'].replace('_', ' ').title() for r in results]
    x = np.arange(len(datasets))
    width = 0.35
    
    # Left panel: RMSE
    ax1 = axes[0]
    baseline_rmse = [r['baseline_rmse'] for r in results]
    sce_rmse = [r['sce_rmse'] for r in results]
    
    # Normalize RMSE for visualization (different scales per dataset)
    # Show relative values as percentage of baseline
    baseline_norm = [100 for _ in results]
    sce_norm = [(r['sce_rmse'] / r['baseline_rmse']) * 100 for r in results]
    
    bars1 = ax1.bar(x - width/2, baseline_norm, width, label='Baseline', 
                    color=COLORS['baseline'], edgecolor='black', linewidth=0.5)
    bars2 = ax1.bar(x + width/2, sce_norm, width, label='+ SCE', 
                    color=COLORS['sce'], edgecolor='black', linewidth=0.5)
    
    ax1.set_ylabel('RMSE (% of Baseline)')
    ax1.set_xlabel('Dataset')
    ax1.set_title('(a) RMSE Reduction', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=15, ha='right')
    ax1.legend(loc='upper right')
    ax1.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax1.set_ylim(0, 110)
    
    # Add improvement percentages
    for i, (b, s) in enumerate(zip(baseline_norm, sce_norm)):
        improvement = results[i]['rmse_improvement_pct']
        ax1.annotate(f'-{improvement:.1f}%', 
                     xy=(x[i] + width/2, s),
                     xytext=(0, 5),
                     textcoords='offset points',
                     ha='center', va='bottom',
                     fontsize=8, color=COLORS['improvement'],
                     fontweight='bold')
    
    # Right panel: R²
    ax2 = axes[1]
    baseline_r2 = [max(r['baseline_r2'], 0) for r in results]  # Clip negative
    sce_r2 = [r['sce_r2'] for r in results]
    
    bars3 = ax2.bar(x - width/2, baseline_r2, width, label='Baseline', 
                    color=COLORS['baseline'], edgecolor='black', linewidth=0.5)
    bars4 = ax2.bar(x + width/2, sce_r2, width, label='+ SCE', 
                    color=COLORS['sce'], edgecolor='black', linewidth=0.5)
    
    ax2.set_ylabel('R² Score')
    ax2.set_xlabel('Dataset')
    ax2.set_title('(b) R² Improvement', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=15, ha='right')
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 1.1)
    
    # Add R² values
    for i, (b, s) in enumerate(zip(baseline_r2, sce_r2)):
        ax2.annotate(f'{s:.2f}', 
                     xy=(x[i] + width/2, s),
                     xytext=(0, 3),
                     textcoords='offset points',
                     ha='center', va='bottom',
                     fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Save in multiple formats
    fig.savefig(output_path.with_suffix('.pdf'))
    fig.savefig(output_path.with_suffix('.png'))
    print(f"  Saved: {output_path.with_suffix('.pdf')}")
    print(f"  Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


def generate_b4_feature_importance(results: list[dict], output_path: Path) -> None:
    """
    Generate B4: Feature Contributions.
    
    Stacked bar chart showing base features vs SCE context features.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    datasets = [r['dataset'].replace('_', ' ').title() for r in results]
    x = np.arange(len(datasets))
    width = 0.6
    
    base_features = [r['n_baseline_features'] for r in results]
    sce_features = [r['n_sce_features'] - r['n_baseline_features'] for r in results]
    
    bars1 = ax.bar(x, base_features, width, label='Base Features', 
                   color=COLORS['baseline'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, sce_features, width, bottom=base_features, 
                   label='SCE Context Features', 
                   color=COLORS['sce'], edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Number of Features')
    ax.set_xlabel('Dataset')
    ax.set_title('Feature Composition: Base vs SCE Context Features', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend(loc='upper right')
    
    # Add total feature counts
    for i, (b, s) in enumerate(zip(base_features, sce_features)):
        total = b + s
        ax.annotate(f'{total}', 
                    xy=(x[i], total),
                    xytext=(0, 3),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    fig.savefig(output_path.with_suffix('.pdf'))
    fig.savefig(output_path.with_suffix('.png'))
    print(f"  Saved: {output_path.with_suffix('.pdf')}")
    print(f"  Saved: {output_path.with_suffix('.png')}")
    plt.close(fig)


def generate_summary_table(results: list[dict], output_path: Path) -> None:
    """
    Generate summary table as both text and LaTeX.
    """
    # Calculate averages
    avg_rmse_impr = sum(r['rmse_improvement_pct'] for r in results) / len(results)
    avg_r2_impr = sum(r['r2_improvement_pct'] for r in results) / len(results)
    
    # Text table
    lines = [
        "=" * 80,
        "EXPERIMENT RESULTS SUMMARY",
        "=" * 80,
        "",
        f"{'Dataset':<25} {'Baseline RMSE':>15} {'SCE RMSE':>12} {'RMSE Δ%':>10} {'R² Δ%':>10}",
        "-" * 80,
    ]
    
    for r in results:
        lines.append(
            f"{r['dataset']:<25} {r['baseline_rmse']:>15,.2f} {r['sce_rmse']:>12,.2f} "
            f"{r['rmse_improvement_pct']:>+9.2f}% {r['r2_improvement_pct']:>+9.2f}%"
        )
    
    lines.extend([
        "-" * 80,
        f"{'AVERAGE':<25} {'':<15} {'':<12} {avg_rmse_impr:>+9.2f}% {avg_r2_impr:>+9.2f}%",
        "=" * 80,
    ])
    
    text_content = "\n".join(lines)
    
    # LaTeX table
    latex_lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{SCE Performance Improvements Across Datasets}",
        r"\label{tab:results}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Dataset & Baseline RMSE & SCE RMSE & RMSE $\Delta$\% \\",
        r"\midrule",
    ]
    
    for r in results:
        name = r['dataset'].replace('_', r'\_')
        latex_lines.append(
            f"{name} & {r['baseline_rmse']:,.0f} & {r['sce_rmse']:,.0f} & "
            f"+{r['rmse_improvement_pct']:.1f}\\% \\\\"
        )
    
    latex_lines.extend([
        r"\midrule",
        f"\\textbf{{Average}} & -- & -- & \\textbf{{+{avg_rmse_impr:.1f}\\%}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    
    latex_content = "\n".join(latex_lines)
    
    # Save files
    with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
        f.write(text_content)
    
    with open(output_path.with_suffix('.tex'), 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"  Saved: {output_path.with_suffix('.txt')}")
    print(f"  Saved: {output_path.with_suffix('.tex')}")
    
    # Also print to console
    print("\n" + text_content)


def main():
    """Generate all publication figures."""
    print("=" * 60)
    print("Generating Publication Figures")
    print("=" * 60)
    
    # Load results
    results = load_results()
    print(f"\nLoaded {len(results)} experiment results")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate figures
    print("\n[B1] Generating consolidated performance figure...")
    generate_b1_performance(results, OUTPUT_DIR / "results_consolidated")
    
    print("\n[B4] Generating feature contributions figure...")
    generate_b4_feature_importance(results, OUTPUT_DIR / "feature_contributions")
    
    print("\n[Summary] Generating results table...")
    generate_summary_table(results, OUTPUT_DIR / "summary_table")
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
