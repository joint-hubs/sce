# Search Experiment Strategies

**Last Updated:** 2026-01-19

---

## Strategy Definitions

### Recommended

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `base_context` | Base + tree-importance-selected context | **Default choice** |
| `base_context_sig_lm` | Base + LM-significant context (p<0.05) | When interpretability matters |

### Full Feature Sets

| Strategy | Description |
|----------|-------------|
| `baseline` | Base features only (no SCE enrichment) |
| `base_context_all` | Base + all context features (300-700) |
| `base_context_sig_tree` | Base + tree-significant context |
| `context_only` | Context features only (no base) |

### Ablation Studies

| Strategy | Purpose |
|----------|---------|
| `ablation_remove_worst_N` | Test robustness to noisy features |
| `ablation_remove_best_N` | Measure contribution of top features |
| `ablation_top_N_only` | Find minimal effective feature set |

### Diagnostics

| Strategy | Purpose |
|----------|---------|
| `context_only` | Verify context alone is insufficient |
| `baseline` | Establish lower bound |

---

## Typical Rankings

The `base_context` strategy consistently ranks **#1** across all datasets tested.

| Rank | Typical Strategy |
|------|------------------|
| 1 | `base_context` |
| 2 | `base_context_sig_tree` |
| 3-5 | `base_context_all` |
| 6+ | `context_only`, `baseline` |

---

## Key Insight

Using all context features (300-700 depending on hierarchy depth) **degrades performance**.
The optimal configuration uses **10-100 selected features** by tree importance.

This suggests:
- Many context statistics are redundant or noisy
- Parsimonious feature selection is essential for optimal SCE performance
- Tree-based importance selection outperforms LM-based (p-value) selection

---

## Configuration

Strategies are defined in `scripts/run.py` in the `run_search_experiment()` function.

```python
strategies = [
    "baseline",
    "base_context",
    "base_context_all",
    "base_context_sig_lm",
    "base_context_sig_tree",
    "context_only",
]
```

Ablation strategies are generated programmatically with varying N values.
