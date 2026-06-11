# Figure Specifications

**Last Updated:** 2026-01-22

---

## Main Paper Tables

### Table 1 — Dataset Overview
- **File:** paper_table1_dataset_overview.{png,pdf,csv,tex}
- **Content:** Dataset summary with n, hierarchy columns, base features, +SCE features
- **Columns:** Dataset, n, #Hier. cols, Base feats, +SCE feats
- **Generator:** `generate_paper_appendix_figures.py`

### Table 2 — Accuracy Summary
- **File:** paper_table2_accuracy_summary.{png,pdf,csv,tex}
- **Content:** RMSE and R² improvements across datasets
- **Columns:** Dataset, Baseline RMSE, +SCE RMSE, ΔRMSE, ΔR²
- **Generator:** `generate_paper_appendix_figures.py`

---

## Main Paper Figures

### M1 — RMSE Improvement
- **File:** paper_m1_rmse_improvement.pdf
- **Content:** Bar chart comparing Baseline vs Best SCE RMSE per dataset
- **Data Table:** paper_m1_rmse_improvement_table.csv
- **Generator:** `generate_paper_appendix_figures.py`

### M2 — Feature Contributions
- **File:** paper_m2_feature_contributions.pdf
- **Content:** Stacked bars showing Base vs Context importance share
- **Data Table:** paper_m2_feature_contributions_table.csv
- **Generator:** `generate_paper_appendix_figures.py`

### M3 — Strategy Ranking
- **File:** paper_m3_strategy_ranking.pdf
- **Content:** Heatmap of strategy ranks across datasets (no grid lines, integer values)
- **Data Table:** paper_m3_strategy_ranking_table.csv
- **Generator:** `generate_paper_appendix_figures.py`

---

## Appendix Figures (Per Dataset)

For each dataset `{rental_poland_long, rental_poland_short, rental_uae_contracts, sales_uae_transactions}`:

### A1 — Baseline vs SCE
- Bar chart comparing baseline and SCE RMSE

### A2 — Strategy Ladder
- Horizontal bar chart of strategies ranked by RMSE

### A3 — Top Features
- Horizontal bar chart of top features, colored by type (Base=blue, SCE=red)

### A4 — Feature Ablation
- Scatter plot showing RMSE vs number of features

### A5 — Context t-Stats
- Histogram of context feature t-statistics from linear model

### A6 — Complexity vs RMSE
- Scatter plot: number of features vs RMSE for all configurations

---

## Output Locations

| Type | Path |
|------|------|
| Main tables | `docs/figures/paper/paper_table*.{png,pdf,csv,tex}` |
| Main figures | `docs/figures/paper/paper_m*.{png,pdf}` |
| Appendix figures | `docs/figures/appendix/appendix_{dataset}_A{1-6}.{png,pdf}` |
| Data tables | `docs/figures/paper/*.csv` |

## Command Reference

```bash
# All figures
python scripts/generate_paper_appendix_figures.py

# Main paper only (M1-M3, Tables 1-2)
python scripts/generate_paper_appendix_figures.py --paper-only

# Appendix only (A1-A6 per dataset)
python scripts/generate_paper_appendix_figures.py --appendix-only

# Tables only
python scripts/generate_paper_appendix_figures.py --tables-only
```
