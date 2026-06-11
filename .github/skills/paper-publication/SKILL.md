---
name: paper-publication
description: ICML 2025 paper visuals, figures, and LaTeX tables. Use when generating figures, formatting tables, or preparing appendix materials. Triggers include figure, M1, M2, M3, publication, LaTeX, consolidated, ICML, paper.
---

# Paper Publication Assets

## Quick Reference

| Document | Purpose |
|---|---|
| references/FIGURE_SPECS.md | Main (M1-M3) and appendix (A1-A6) figure specs |
| references/LATEX_TABLES.md | Table templates |

## When to Use This Skill

- You need to regenerate consolidated figures
- You need to update LaTeX tables from new results
- You need to generate Table 1 (dataset overview) or Table 2 (accuracy summary)

## Figure & Table Naming

| Output | Content | Location |
|--------|---------|----------|
| Table 1 | Dataset overview (n, hierarchy cols, features) | paper/paper_table1_dataset_overview.* |
| Table 2 | Accuracy summary (RMSE, R², improvements) | paper/paper_table2_accuracy_summary.* |
| M1 | RMSE Improvement (bar chart) | paper/paper_m1_rmse_improvement.* |
| M2 | Feature Contributions (stacked bars) | paper/paper_m2_feature_contributions.* |
| M3 | Strategy Ranking (heatmap, no grid lines) | paper/paper_m3_strategy_ranking.* |
| A1-A6 | Per-dataset appendix figures | appendix/appendix_{dataset}_A{1-6}.* |

## Procedure

1. Run search experiments to generate latest results (`python scripts/run.py --all --search`).
2. Run `generate_paper_appendix_figures.py` to create figures.
3. Use flags: `--paper-only`, `--appendix-only`, `--tables-only` as needed.
4. Export PDF/PNG for the paper (both formats generated automatically).
5. LaTeX tables (.tex) are also generated alongside figures.

## Script Usage

```bash
# Generate all figures (paper + appendix)
python scripts/generate_paper_appendix_figures.py

# Generate only main paper figures (M1-M3, Tables 1-2)
python scripts/generate_paper_appendix_figures.py --paper-only

# Generate only appendix figures (A1-A6 per dataset)
python scripts/generate_paper_appendix_figures.py --appendix-only

# Generate only Tables 1 and 2
python scripts/generate_paper_appendix_figures.py --tables-only
```

## Output Directories

- `docs/figures/paper/` - Main paper figures (M1-M3, Tables 1-2)
- `docs/figures/appendix/` - Per-dataset appendix figures

## SCE Feature Classification

Features are classified as **Base** (blue) or **SCE Context** (red) based on suffixes:
- SCE suffixes: `_mean`, `_median`, `_std`, `_fold_lower`, `_fold_upper`, `_q05`, `_q10`, etc.
- The `is_sce_feature()` function handles both simple and compound suffixes.
