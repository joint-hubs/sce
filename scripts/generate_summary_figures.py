#!/usr/bin/env python3
"""
@module: scripts.generate_summary_figures
@depends: results/categorical_mode_batch_summary_*, results/experiment_results.json
@exports: Publication-quality summary figures comparing SCE vs Baseline
@paper_ref: Main results figures

Generates professional, ICML-grade summary visualizations of SCE results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

# ── Publication-quality style ──────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.2,
})

# ── Color palette (colorblind-safe, publication-ready) ─────────────────────
PALETTE = {
    "baseline":     "#6C7A89",   # cool grey
    "sce":          "#1B9E77",   # teal-green (ColorBrewer Dark2)
    "improvement":  "#D95F02",   # burnt orange
    "negative":     "#E7298A",   # magenta-pink
    "accent":       "#7570B3",   # muted purple
    "bg_band":      "#F7F7F7",   # very light grey for alternating rows
    "grid":         "#CCCCCC",
}

DATASET_LABELS = {
    "rental_poland_short": "Airbnb Poland",
    "melbourne_housing": "Melbourne Housing",
    "m5_store_dept_daily": "M5 Demand",
    "walmart_weekly": "Walmart Weekly",
    "rossmann_daily": "Rossmann Daily",
}

MODEL_ORDER = [
    "CatBoost", "XGBoost", "LightGBM",
    "Gradient Boosting", "Random Forest", "Extra Trees", "Ridge",
]

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = ROOT / "docs" / "figures" / "paper"


# ── Data loading ───────────────────────────────────────────────────────────

def load_cross_model_matrix() -> pd.DataFrame:
    """Load the latest categorical_mode_matrix.csv."""
    batch_dirs = sorted(RESULTS_DIR.glob("categorical_mode_batch_summary_*"))
    if not batch_dirs:
        raise FileNotFoundError("No categorical mode batch summary found.")
    latest = batch_dirs[-1]
    csv_path = latest / "categorical_mode_matrix.csv"
    df = pd.read_csv(csv_path)
    # Use the best of manual/auto SCE per row
    df["best_sce_rmse"] = df[["manual_sce_rmse", "auto_sce_rmse"]].min(axis=1)
    df["best_sce_r2"] = df[["manual_sce_r2", "auto_sce_r2"]].max(axis=1)
    df["best_rmse_improvement_pct"] = (
        (df["baseline_rmse"] - df["best_sce_rmse"]) / df["baseline_rmse"] * 100
    )
    df["best_r2_improvement_pp"] = (df["best_sce_r2"] - df["baseline_r2"]) * 100
    df["display_dataset"] = df["dataset"].map(DATASET_LABELS)
    return df


def load_experiment_results() -> pd.DataFrame:
    """Load the simple experiment_results.json for high-level summary."""
    path = RESULTS_DIR / "experiment_results.json"
    with open(path) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def best_per_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """For each dataset, pick the model with the best SCE RMSE improvement %."""
    # Exclude Ridge (often degrades) from "best" selection for summary
    df_clean = df[df["model_label"] != "Ridge"].copy()
    idx = df_clean.groupby("dataset")["best_rmse_improvement_pct"].idxmax()
    return df_clean.loc[idx].sort_values("best_rmse_improvement_pct", ascending=False)


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"{name}.{ext}")
    plt.close(fig)


# ── Figure 1: RMSE Reduction Summary (paired bars + % annotations) ────────

def fig1_rmse_reduction(df: pd.DataFrame) -> None:
    """Paired horizontal bars: Baseline vs Best SCE, sorted by improvement %."""
    best = best_per_dataset(df)
    best = best.sort_values("best_rmse_improvement_pct", ascending=True)  # ascending for horizontal

    datasets = best["display_dataset"].values
    baseline = best["baseline_rmse"].values
    sce = best["best_sce_rmse"].values
    improvement = best["best_rmse_improvement_pct"].values

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    y = np.arange(len(datasets))
    bar_h = 0.32

    ax.barh(y + bar_h / 2, baseline, bar_h,
            color=PALETTE["baseline"], edgecolor="white", linewidth=0.5,
            label="Baseline", zorder=3)
    ax.barh(y - bar_h / 2, sce, bar_h,
            color=PALETTE["sce"], edgecolor="white", linewidth=0.5,
            label="+ SCE (best)", zorder=3)

    # % improvement annotations to the right
    x_max = max(baseline) * 1.35
    for i, (b, s, imp) in enumerate(zip(baseline, sce, improvement)):
        ax.annotate(
            f"$\\downarrow${imp:.1f}%",
            xy=(max(b, s) * 1.02, y[i]),
            fontsize=8.5, fontweight="bold",
            color=PALETTE["improvement"],
            va="center",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=9)
    ax.set_xlabel("RMSE")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_xlim(right=x_max)
    ax.legend(loc="lower right", frameon=True)
    ax.set_title("RMSE Reduction: Baseline vs SCE-Enhanced")

    # Light alternating row bands
    for i in range(len(datasets)):
        if i % 2 == 0:
            ax.axhspan(y[i] - 0.5, y[i] + 0.5, color=PALETTE["bg_band"], zorder=0)

    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    fig.tight_layout()
    save(fig, "summary_fig1_rmse_reduction")


# ── Figure 2: Cross-Model RMSE Improvement Heatmap ────────────────────────

def fig2_cross_model_heatmap(df: pd.DataFrame) -> None:
    """Heatmap: Model x Dataset with RMSE improvement %, diverging colormap."""
    # Exclude Ridge (extreme outlier distorts the color scale)
    df_plot = df[df["model_label"] != "Ridge"].copy()

    # Pivot to matrix form
    pivot = df_plot.pivot_table(
        index="model_label", columns="display_dataset",
        values="best_rmse_improvement_pct",
    )
    # Reorder
    dataset_order = [DATASET_LABELS[k] for k in DATASET_LABELS if DATASET_LABELS[k] in pivot.columns]
    model_order = [m for m in MODEL_ORDER if m in pivot.index]
    pivot = pivot.reindex(index=model_order, columns=dataset_order)

    # Custom diverging colormap: pink (negative) -> white -> teal (positive)
    cmap = LinearSegmentedColormap.from_list(
        "sce_div", ["#E7298A", "#FFFFFF", "#1B9E77"], N=256
    )
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)].min()),
               abs(pivot.values[~np.isnan(pivot.values)].max()))
    vmax = min(vmax, 30)  # cap for visual clarity

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto",
                   vmin=-vmax, vmax=vmax)

    # Text annotations
    for i in range(len(model_order)):
        for j in range(len(dataset_order)):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > vmax * 0.65 else "black"
            sign = "+" if val > 0 else ""
            ax.text(j, i, f"{sign}{val:.1f}%", ha="center", va="center",
                    fontsize=7.5, color=color, fontweight="medium")

    ax.set_xticks(range(len(dataset_order)))
    ax.set_xticklabels(dataset_order, rotation=25, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order, fontsize=8.5)

    # Remove spines from heatmap
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("RMSE Improvement (%)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)

    ax.set_title("RMSE Improvement (%) by Model and Dataset")
    fig.tight_layout()
    save(fig, "summary_fig2_cross_model_heatmap")


# ── Figure 3: R-squared Improvement Dot Plot ──────────────────────────────

def fig3_r2_dot_plot(df: pd.DataFrame) -> None:
    """Dot plot of R2 improvement (pp) per dataset, showing model spread."""
    df_plot = df[df["model_label"] != "Ridge"].copy()

    datasets = [DATASET_LABELS[k] for k in DATASET_LABELS]
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    for i, ds_label in enumerate(datasets):
        subset = df_plot[df_plot["display_dataset"] == ds_label]
        vals = subset["best_r2_improvement_pp"].values
        med = np.median(vals)

        # Individual model dots (jittered slightly)
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        colors = [PALETTE["sce"] if v >= 0 else PALETTE["negative"] for v in vals]
        ax.scatter(vals, np.full_like(vals, i) + jitter,
                   c=colors, s=40, alpha=0.7, edgecolors="white", linewidths=0.4, zorder=4)

        # Median marker
        ax.scatter([med], [i], marker="D", s=60, c=PALETTE["improvement"],
                   edgecolors="black", linewidths=0.5, zorder=5)

        # Interquartile range bar
        q25, q75 = np.percentile(vals, [25, 75])
        ax.plot([q25, q75], [i, i], color=PALETTE["improvement"],
                linewidth=2, solid_capstyle="round", zorder=3)

    ax.axvline(0, color="black", linewidth=0.5, linestyle="-", zorder=2)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=9)
    ax.set_xlabel("$\\Delta R^2$ (percentage points)")
    ax.set_title("$R^2$ Improvement Distribution Across Models")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    # Alternating bands
    for i in range(len(datasets)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=PALETTE["bg_band"], zorder=0)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=PALETTE["sce"],
               markersize=6, label="Individual models"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=PALETTE["improvement"],
               markeredgecolor="black", markersize=6, label="Median"),
        Line2D([0], [0], color=PALETTE["improvement"], linewidth=2, label="IQR"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7, frameon=True)

    fig.tight_layout()
    save(fig, "summary_fig3_r2_improvement")


# ── Figure 4: Consolidated 3-panel dashboard ──────────────────────────────

def fig4_dashboard(df: pd.DataFrame) -> None:
    """3-panel consolidated figure for the main paper body."""
    df_plot = df[df["model_label"] != "Ridge"].copy()
    best = best_per_dataset(df)
    best = best.sort_values("best_rmse_improvement_pct", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5),
                             gridspec_kw={"width_ratios": [1.1, 1.3, 1.0]})

    # ── Panel A: RMSE % improvement bars ──────────────────────────────
    ax = axes[0]
    datasets = best["display_dataset"].values
    improvements = best["best_rmse_improvement_pct"].values
    y = np.arange(len(datasets))

    bars = ax.barh(y, improvements, color=PALETTE["sce"], edgecolor="white",
                   linewidth=0.5, height=0.6, zorder=3)
    for i, (val, bar) in enumerate(zip(improvements, bars)):
        ax.text(val + 0.3, y[i], f"{val:.1f}%", va="center", fontsize=8,
                fontweight="bold", color=PALETTE["sce"])
    ax.set_yticks(y)
    ax.set_yticklabels(datasets, fontsize=8.5)
    ax.set_xlabel("RMSE Reduction (%)")
    ax.set_title("(a) Best RMSE Improvement", fontsize=10)
    ax.set_xlim(0, max(improvements) * 1.25)
    ax.grid(axis="x", linestyle="--", alpha=0.3, zorder=0)
    for i in range(len(datasets)):
        if i % 2 == 0:
            ax.axhspan(y[i] - 0.5, y[i] + 0.5, color=PALETTE["bg_band"], zorder=0)

    # ── Panel B: Cross-model heatmap ──────────────────────────────────
    ax = axes[1]
    pivot = df_plot.pivot_table(
        index="model_label", columns="display_dataset",
        values="best_rmse_improvement_pct",
    )
    dataset_order = [DATASET_LABELS[k] for k in DATASET_LABELS if DATASET_LABELS[k] in pivot.columns]
    model_order_local = [m for m in MODEL_ORDER if m in pivot.index]
    pivot = pivot.reindex(index=model_order_local, columns=dataset_order)

    cmap = LinearSegmentedColormap.from_list(
        "sce_div", ["#E7298A", "#FFFFFF", "#1B9E77"], N=256
    )
    vals = pivot.values[~np.isnan(pivot.values)]
    vmax = min(max(abs(vals.min()), abs(vals.max())), 25)

    im = ax.imshow(pivot.values, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)
    for i in range(len(model_order_local)):
        for j in range(len(dataset_order)):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            color = "white" if abs(val) > vmax * 0.65 else "black"
            sign = "+" if val > 0 else ""
            ax.text(j, i, f"{sign}{val:.0f}", ha="center", va="center",
                    fontsize=7, color=color)

    short_labels = [d.split()[-1] for d in dataset_order]  # last word only
    ax.set_xticks(range(len(dataset_order)))
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(model_order_local)))
    ax.set_yticklabels(model_order_local, fontsize=7.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("ΔRMSE %", fontsize=7.5)
    ax.set_title("(b) RMSE Improvement by Model", fontsize=10)

    # ── Panel C: R2 improvement dot plot ──────────────────────────────
    ax = axes[2]
    for i, ds_label in enumerate(dataset_order):
        subset = df_plot[df_plot["display_dataset"] == ds_label]
        vals = subset["best_r2_improvement_pp"].values
        med = np.median(vals)
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        colors = [PALETTE["sce"] if v >= 0 else PALETTE["negative"] for v in vals]
        ax.scatter(vals, np.full_like(vals, i) + jitter,
                   c=colors, s=25, alpha=0.7, edgecolors="white", linewidths=0.3, zorder=4)
        ax.scatter([med], [i], marker="D", s=40, c=PALETTE["improvement"],
                   edgecolors="black", linewidths=0.4, zorder=5)
        q25, q75 = np.percentile(vals, [25, 75])
        ax.plot([q25, q75], [i, i], color=PALETTE["improvement"],
                linewidth=1.8, solid_capstyle="round", zorder=3)

    ax.axvline(0, color="black", linewidth=0.5, zorder=2)
    short_ds = [d.split()[-1] for d in dataset_order]
    ax.set_yticks(range(len(dataset_order)))
    ax.set_yticklabels(short_ds, fontsize=7.5)
    ax.set_xlabel("$\\Delta R^2$ (pp)", fontsize=8.5)
    ax.set_title("(c) $R^2$ Improvement", fontsize=10)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    for i in range(len(dataset_order)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=PALETTE["bg_band"], zorder=0)

    fig.tight_layout(w_pad=2.0)
    save(fig, "summary_fig4_dashboard")


# ── Figure 5: Win-rate bar chart (model robustness) ───────────────────────

def fig5_model_win_rates(df: pd.DataFrame) -> None:
    """Stacked bar: per model, how often SCE improves vs degrades RMSE."""
    df_plot = df[df["model_label"] != "Ridge"].copy()
    models = [m for m in MODEL_ORDER if m in df_plot["model_label"].values]

    wins = []
    losses = []
    for m in models:
        subset = df_plot[df_plot["model_label"] == m]
        w = (subset["best_rmse_improvement_pct"] > 0).sum()
        l = (subset["best_rmse_improvement_pct"] <= 0).sum()
        wins.append(w)
        losses.append(l)

    fig, ax = plt.subplots(figsize=(5.0, 2.8))
    y = np.arange(len(models))
    total = np.array(wins) + np.array(losses)

    ax.barh(y, wins, color=PALETTE["sce"], edgecolor="white",
            linewidth=0.5, height=0.55, label="SCE improves RMSE", zorder=3)
    ax.barh(y, losses, left=wins, color=PALETTE["negative"], edgecolor="white",
            linewidth=0.5, height=0.55, label="SCE degrades RMSE", zorder=3)

    for i in range(len(models)):
        pct = wins[i] / total[i] * 100 if total[i] > 0 else 0
        ax.text(total[i] + 0.1, y[i], f"{pct:.0f}%", va="center",
                fontsize=8, fontweight="bold", color=PALETTE["sce"])

    ax.set_yticks(y)
    ax.set_yticklabels(models, fontsize=8.5)
    ax.set_xlabel("Number of datasets")
    ax.set_title("SCE Win Rate by Model (excl. Ridge)")
    ax.legend(loc="lower right", fontsize=7, frameon=True)
    ax.set_xlim(0, max(total) + 1.2)

    for i in range(len(models)):
        if i % 2 == 0:
            ax.axhspan(y[i] - 0.5, y[i] + 0.5, color=PALETTE["bg_band"], zorder=0)

    fig.tight_layout()
    save(fig, "summary_fig5_win_rates")


# ── Figure 6: Summary table figure ────────────────────────────────────────

def fig6_summary_table(df: pd.DataFrame) -> None:
    """Publication-quality table rendered as a figure."""
    df_plot = df[df["model_label"] != "Ridge"].copy()

    rows = []
    for ds_key, ds_label in DATASET_LABELS.items():
        subset = df_plot[df_plot["dataset"] == ds_key]
        if subset.empty:
            continue
        # Best model for this dataset
        best_idx = subset["best_rmse_improvement_pct"].idxmax()
        best_row = subset.loc[best_idx]

        # Aggregate stats across all (non-Ridge) models
        mean_imp = subset["best_rmse_improvement_pct"].mean()
        median_r2 = subset["best_r2_improvement_pp"].median()
        n_wins = (subset["best_rmse_improvement_pct"] > 0).sum()
        n_total = len(subset)

        rows.append({
            "Dataset": ds_label,
            "n": f"{int(best_row.get('baseline_rmse', 0)):,}".replace(",", ""),  # placeholder
            "Baseline\nRMSE": f"{best_row['baseline_rmse']:,.0f}",
            "Best SCE\nRMSE": f"{best_row['best_sce_rmse']:,.0f}",
            "ΔRMSE": f"↓{best_row['best_rmse_improvement_pct']:.1f}%",
            "Mean ΔRMSE\n(all models)": f"↓{mean_imp:.1f}%" if mean_imp > 0 else f"↑{-mean_imp:.1f}%",
            "Median ΔR²\n(pp)": f"+{median_r2:.1f}" if median_r2 >= 0 else f"{median_r2:.1f}",
            "Win Rate": f"{n_wins}/{n_total}",
        })

    table_df = pd.DataFrame(rows)
    # Drop the 'n' column (placeholder) — use only meaningful columns
    table_df = table_df.drop(columns=["n"])

    n_cols = len(table_df.columns)
    n_rows = len(table_df)
    fig_width = 9.5
    fig_height = 1.2 + 0.4 * n_rows

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.auto_set_column_width(col=list(range(n_cols)))
    table.scale(1, 1.6)

    # Style header row
    for j in range(n_cols):
        cell = table[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        cell.set_edgecolor("white")

    # Alternating row colors
    for i in range(1, n_rows + 1):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_edgecolor("#E0E0E0")
            if i % 2 == 0:
                cell.set_facecolor("#F2F7FB")
            else:
                cell.set_facecolor("white")

    ax.set_title("Summary of SCE Improvement Across Datasets",
                 fontsize=11, fontweight="bold", pad=20)
    fig.tight_layout()
    save(fig, "summary_fig6_table")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading cross-model results...")
    df = load_cross_model_matrix()
    print(f"  Loaded {len(df)} model×dataset combinations.")

    print("\nGenerating figures:")

    print("  [1/6] RMSE Reduction Summary...")
    fig1_rmse_reduction(df)

    print("  [2/6] Cross-Model Heatmap...")
    fig2_cross_model_heatmap(df)

    print("  [3/6] R² Improvement Dot Plot...")
    fig3_r2_dot_plot(df)

    print("  [4/6] Consolidated Dashboard...")
    fig4_dashboard(df)

    print("  [5/6] Model Win Rates...")
    fig5_model_win_rates(df)

    print("  [6/6] Summary Table...")
    fig6_summary_table(df)

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("  summary_fig1_rmse_reduction.{png,pdf}")
    print("  summary_fig2_cross_model_heatmap.{png,pdf}")
    print("  summary_fig3_r2_improvement.{png,pdf}")
    print("  summary_fig4_dashboard.{png,pdf}")
    print("  summary_fig5_win_rates.{png,pdf}")
    print("  summary_fig6_table.{png,pdf}")


if __name__ == "__main__":
    main()
