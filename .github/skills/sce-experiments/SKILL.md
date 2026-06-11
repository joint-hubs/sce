---
name: sce-experiments
description: Running SCE experiments and reproducing paper results. Use when running analyses, reproducing results, configuring experiments, interpreting metrics, or batch processing. Triggers include run, experiment, reproduce, results, RMSE, R2, baseline, enriched.
---

# SCE Experiments

## Quick Reference

| Item | Location |
|---|---|
| Run script | scripts/run.py |
| Batch summary script | scripts/generate_search_batch_summary.py |
| Configs | configs/ |
| Results | results/ |

## When to Use This Skill

- You need to reproduce a result from the paper
- You need to run the full benchmark suite
- You need to interpret RMSE and R2 changes
- You need to run search experiments

---

## Search Experiments

The `run_search_experiment()` function in `scripts/run.py` runs a comprehensive 
strategy comparison across multiple feature configurations.

### Running a Search Experiment

```bash
python scripts/run.py --config configs/rental_poland_short.toml --mode search
```

### Strategies Tested

| Strategy | Description |
|----------|-------------|
| `baseline` | Base features only (no SCE) |
| `base_context` | Base + selected context features **(BEST)** |
| `base_context_all` | Base + all context features |
| `base_context_sig_lm` | Base + LM-significant context |
| `base_context_sig_tree` | Base + tree-significant context |
| `context_only` | Context features only |
| `ablation_remove_worst_N` | Remove N worst features |
| `ablation_remove_best_N` | Remove N best features |
| `ablation_top_N_only` | Keep only top N features |

### Key Finding

**Feature selection is critical.** Using all 300-700 context features degrades performance.
The optimal configuration uses 10-100 selected context features.

### Output Files

Results are saved to `results/{dataset}_search_{timestamp}/data/`:
- `model_comparison.csv` - All configurations tested
- `best_by_strategy.csv` - Best config per strategy
- `strategy_comparison.md` - Human-readable summary
- `aggregated_feature_importance.csv` - Feature importance across models

To summarize an explicit set of completed search runs into combined tables and plots:

```bash
python scripts/generate_search_batch_summary.py \
	results/run_a_search_YYYYMMDD_HHMMSS \
	results/run_b_search_YYYYMMDD_HHMMSS \
	--output-dir results/search_batch_summary_YYYYMMDD_HHMMSS
```

### Current Results (2026-01-19)

| Dataset | Baseline RMSE | Best RMSE | Strategy | Features | Improvement |
|---------|---------------|-----------|----------|----------|-------------|
| Poland Long | 4,580.89 | 4,541.05 | base_context | 10 | +0.87% |
| Poland Short | 27,367.70 | 22,541.28 | base_context | 37 | +17.64% |
| UAE Contracts | 465,037.04 | 360,266.87 | base_context | 88 | +22.53% |
| UAE Transactions | 32,489,660 | 26,353,228 | base_context | 31 | +18.89% |

---

## Procedure

1. Confirm dataset availability (local or remote).
2. Run the script with appropriate config.
3. Validate outputs match expected ranges.
4. Export summary for README/paper tables.

---

## Resources

- [Search Strategies Reference](references/SEARCH_STRATEGIES.md)
- [Results 2026-01-19](references/RESULTS_2026-01-19.md)

---

## Gotchas

- Ensure cross-fitting (if enabled) is deterministic.
- Log seeds, config hashes, and git SHA.
- Windows consoles using cp1250 can choke on Unicode R2; use ASCII R2 in logs.
- SCE refuses to run when categorical columns are below `min_categorical_columns`.
- Search runs default to 50k samples for UAE/Dubai datasets to avoid OOM.
- `results/` is gitignored; if you need to push run outputs, adjust `.gitignore`.
- `lm_context_statistics.csv` reports `t_statistic` values (can be around -6); don't confuse these with model `r2` from `model_comparison.csv`.
- Temporal or asymmetric splits can drop zero-variance base columns on the test side; search and pruning now align to the train/test column intersection, so any custom analysis code should do the same.
