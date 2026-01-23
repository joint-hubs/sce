# Experiments

This document describes how to reproduce the SCE validation experiments and generate the paper figures.

## Quick Start

`ash
# Install with data dependencies
pip install -e ".[data]"

# Run all experiments
python scripts/run.py --all

# Generate figures
python scripts/run.py --generate-figures
`

## Datasets

SCE was validated on 4 real-world property pricing datasets:

| Dataset | Records | Description |
|---------|---------|-------------|
| rental_poland_short | 830 | Poland apartment rentals (short-term, Airbnb) |
| rental_poland_long | 1,005 | Poland apartment rentals (long-term, Otodom) |
| rental_uae_contracts | 19,799 | UAE rental contracts (Dubai) |
| sales_uae_transactions | 19,800 | UAE property sales (Dubai) |

### Download Datasets

Poland datasets are included in `data/parquet/`. UAE datasets require download:

`ash
python scripts/download_datasets.py
`

## Experiment Commands

### Run Individual Dataset

`ash
# Run experiment for a specific dataset
python scripts/run.py --dataset rental_poland_short
python scripts/run.py --dataset rental_poland_long
python scripts/run.py --dataset rental_uae_contracts
python scripts/run.py --dataset sales_uae_transactions
`

### Run All Experiments

`ash
python scripts/run.py --all
`

This will:
1. Load each dataset from `data/parquet/`
2. Run baseline XGBoost model
3. Run XGBoost + SCE context features
4. Compare RMSE and R metrics
5. Save results to `results/`

### Generate Paper Figures

After running experiments:

`ash
python scripts/run.py --generate-figures
`

Or for detailed appendix figures:

`ash
python scripts/generate_paper_appendix_figures.py
`

## Results

### Main Paper Figures

| Figure | Description | File |
|--------|-------------|------|
| M1 | RMSE Improvement | ![M1](figures/paper/paper_m1_rmse_improvement.png) |
| M2 | Feature Contributions | ![M2](figures/paper/paper_m2_feature_contributions.png) |
| M3 | Strategy Ranking | ![M3](figures/paper/paper_m3_strategy_ranking.png) |

### Key Results (Summary)

| Dataset | Baseline RMSE | + SCE RMSE | ΔRMSE | ΔR² |
|---------|---------------|------------|-------|-----|
| rental_poland_long | 4,581 | 4,541 | ↓ 0.9% | +1.55 pp |
| rental_poland_short | 27,368 | 22,541 | ↓ 17.6% | +24.49 pp |
| rental_uae_contracts | 465,037 | 360,267 | ↓ 22.5% | +3.83 pp |
| sales_uae_transactions | 32,489,660 | 26,353,228 | ↓ 18.9% | +25.83 pp |

**Average RMSE improvement: ~15%**

### Appendix Figures (Per-Dataset)

Each dataset has 6 appendix figures (A1-A6):

- **A1**: RMSE comparison (baseline vs SCE)
- **A2**: R comparison (baseline vs SCE)
- **A3**: Feature importance breakdown
- **A4**: Hierarchy level contributions
- **A5**: Cross-validation stability
- **A6**: Residual analysis

| Dataset | Figures |
|---------|---------|
| rental_poland_short | [A1](figures/appendix/appendix_rental_poland_short_A1.png) [A2](figures/appendix/appendix_rental_poland_short_A2.png) [A3](figures/appendix/appendix_rental_poland_short_A3.png) [A4](figures/appendix/appendix_rental_poland_short_A4.png) [A5](figures/appendix/appendix_rental_poland_short_A5.png) [A6](figures/appendix/appendix_rental_poland_short_A6.png) |
| rental_poland_long | [A1](figures/appendix/appendix_rental_poland_long_A1.png) [A2](figures/appendix/appendix_rental_poland_long_A2.png) [A3](figures/appendix/appendix_rental_poland_long_A3.png) [A4](figures/appendix/appendix_rental_poland_long_A4.png) [A5](figures/appendix/appendix_rental_poland_long_A5.png) [A6](figures/appendix/appendix_rental_poland_long_A6.png) |
| rental_uae_contracts | [A1](figures/appendix/appendix_rental_uae_contracts_A1.png) [A2](figures/appendix/appendix_rental_uae_contracts_A2.png) [A3](figures/appendix/appendix_rental_uae_contracts_A3.png) [A4](figures/appendix/appendix_rental_uae_contracts_A4.png) [A5](figures/appendix/appendix_rental_uae_contracts_A5.png) [A6](figures/appendix/appendix_rental_uae_contracts_A6.png) |
| sales_uae_transactions | [A1](figures/appendix/appendix_sales_uae_transactions_A1.png) [A2](figures/appendix/appendix_sales_uae_transactions_A2.png) [A3](figures/appendix/appendix_sales_uae_transactions_A3.png) [A4](figures/appendix/appendix_sales_uae_transactions_A4.png) [A5](figures/appendix/appendix_sales_uae_transactions_A5.png) [A6](figures/appendix/appendix_sales_uae_transactions_A6.png) |

## Reproducibility Notes

- **Random seed**: 42 (set in all config files)
- **Cross-validation**: 5-fold for leakage-safe context
- **Train/test split**: 80/20
- **XGBoost**: `n_estimators=500`, `max_depth=6`, `learning_rate=0.05`

All experiments were run on Python 3.11 with:
- scikit-learn 1.4.0
- xgboost 2.0.3
- pandas 2.1.4
- numpy 1.26.3
