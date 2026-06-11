# SCE Audit Suggestions  2026-01-19

**Auditor:** GitHub Copilot (Auditor Mode)  
**Scope:** Skills, References, Paper Enhancements

---

## 1. Skills Requiring Updates

### 1.1 sce-algorithm/SKILL.md
**File:** .github/skills/sce-algorithm/SKILL.md

| Issue | Current State | Recommended Fix |
|-------|---------------|-----------------|
| Outdated results in "Current Status" | Claims +62%, +97%, +65% improvements | Update to current: +0.87%, +17.6%, +22.5%, +18.9% |
| Misleading success metrics | Results from leaky configuration | Clarify these are **search experiment** results with feature selection |

**Specific text to update (around line 45):**
`markdown
# OLD (REMOVE):
Results with correct methodology:
- Poland short-term rentals: +62% RMSE improvement, R 0.006  0.857
- Poland long-term rentals: +97% RMSE improvement, R 0.037  0.999  
- UAE rental contracts: +65% RMSE improvement, R 0.940  0.993

# NEW:
**Current Results (Search Experiments 2026-01-19):**
| Dataset | Baseline RMSE | Best SCE RMSE | Improvement | Features |
|---------|---------------|---------------|-------------|----------|
| Poland Long | 4,581 | 4,541 | +0.87% | 10 |
| Poland Short | 27,368 | 22,541 | +17.64% | 37 |
| UAE Contracts | 465,037 | 360,267 | +22.53% | 88 |
| UAE Transactions | 32,489,660 | 26,353,228 | +18.89% | 31 |

*Note: Best results use ase_context strategy with feature selection.*
`

---

### 1.2 sce-algorithm/references/CODE_MAP.md
**File:** .github/skills/sce-algorithm/references/CODE_MAP.md

| Issue | Location | Fix |
|-------|----------|-----|
| Outdated "Verified Results" table | Lines 215-220 | Update with current search experiment results |
| Mentions +56% improvement | Line ~95 (backoff section) | Remove or update |

---

### 1.3 sce-experiments/SKILL.md
**File:** .github/skills/sce-experiments/SKILL.md

**Missing content to add:**

`markdown
## Search Experiments

The `run_search_experiment()` function in `scripts/run.py` runs a comprehensive 
strategy comparison across multiple feature configurations.

### Strategies Tested

| Strategy | Description |
|----------|-------------|
| aseline | Base features only (no SCE) |
| ase_context | Base + selected context features (BEST) |
| ase_context_all | Base + all context features |
| ase_context_sig_lm | Base + LM-significant context |
| ase_context_sig_tree | Base + tree-significant context |
| context_only | Context features only |
| blation_remove_worst_N | Remove N worst features |
| blation_remove_best_N | Remove N best features |
| blation_top_N_only | Keep only top N features |

### Key Finding

**Feature selection is critical.** Using all 300-700 context features degrades performance.
The optimal configuration uses 10-100 selected context features.

### Running Search Experiments

`ash
python scripts/run.py --config configs/rental_poland_short.toml --mode search
`

### Output Files

Results are saved to `results/{dataset}_search_{timestamp}/data/`:
- model_comparison.csv  All configurations tested
- est_by_strategy.csv  Best config per strategy
- strategy_comparison.md  Human-readable summary
- ggregated_feature_importance.csv  Feature importance across models
`

---

### 1.4 paper-publication/SKILL.md
**File:** .github/skills/paper-publication/SKILL.md

| Issue | Fix |
|-------|-----|
| References B1/B2/B4 figure names | Update to M1/M2/M3 naming |
| Missing appendix figure specs | Add A1-A6 per-dataset specs |

---

### 1.5 paper-publication/references/FIGURE_SPECS.md
**File:** .github/skills/paper-publication/references/FIGURE_SPECS.md

**Replace with:**

`markdown
# Figure Specifications

## Main Paper Figures

### M1  RMSE Improvement
- File: paper_m1_rmse_improvement.pdf
- Content: Bar chart comparing Baseline vs Best SCE RMSE per dataset
- Table: paper_m1_rmse_improvement_table.csv

### M2  Feature Contributions
- File: paper_m2_feature_contributions.pdf
- Content: Stacked bars showing Base vs Context importance share
- Table: paper_m2_feature_contributions_table.csv

### M3  Strategy Ranking
- File: paper_m3_strategy_ranking.pdf
- Content: Heatmap of strategy ranks across datasets
- Table: paper_m3_strategy_ranking_table.csv

## Appendix Figures (Per Dataset)

For each dataset {rental_poland_long, rental_poland_short, rental_uae_contracts, sales_uae_transactions}:

### A1  RMSE by Strategy
### A2  R by Strategy
### A3  Feature Count vs Performance
### A4  Top 20 Feature Importance
### A5  Pruning Trace
### A6  Context Feature Distribution
`

---

## 2. New References to Add

### 2.1 sce-algorithm/references/AUDIT_2026-01-19.md (NEW)
**Create:** .github/skills/sce-algorithm/references/AUDIT_2026-01-19.md

`markdown
# Scientific Audit  2026-01-19

## Summary
All equations (1-4), Algorithm 1, and hierarchical backoff verified against production code.

## Equation Verification

| Equation | Implementation | Status |
|----------|----------------|--------|
| Eq. 1: Context Vector | StatsAggregator.aggregate() |  |
| Eq. 2: Concatenation | compute_aggregations() + 	ransform() |  |
| Eq. 3: Relative Features | compute_relative_features()  **DISABLED** |  |
| Eq. 4: Cross-fitting | _fit_transform_cross_fitted() |  |

## Leakage Prevention
- Cross-fitting:  KFold(n_splits=5, shuffle=True, random_state=42)
- Relative features:  Disabled by default (causes y_t leakage)
- Global stats:  Computed out-of-fold

## Current Results
See search experiment outputs in `results/*_search_*/`
`

---

### 2.2 sce-experiments/references/SEARCH_STRATEGIES.md (NEW)
**Create:** .github/skills/sce-experiments/references/SEARCH_STRATEGIES.md

`markdown
# Search Experiment Strategies

## Strategy Definitions

### Recommended
| Strategy | Description | When to Use |
|----------|-------------|-------------|
| ase_context | Base + tree-importance-selected context | **Default choice** |
| ase_context_sig_lm | Base + LM-significant context (p<0.05) | When interpretability matters |

### Ablation
| Strategy | Purpose |
|----------|---------|
| blation_remove_worst_N | Test robustness to noisy features |
| blation_remove_best_N | Measure contribution of top features |
| blation_top_N_only | Find minimal effective feature set |

### Diagnostics
| Strategy | Purpose |
|----------|---------|
| context_only | Verify context alone is insufficient |
| aseline | Establish lower bound |

## Typical Rankings

The ase_context strategy consistently ranks #1 across all datasets tested.
Using all features (ase_context_all) typically ranks #3-5.
`

---

### 2.3 sce-experiments/references/RESULTS_2026-01-19.md (NEW)
**Create:** .github/skills/sce-experiments/references/RESULTS_2026-01-19.md

`markdown
# Search Experiment Results  2026-01-19

## Summary

| Dataset | Baseline RMSE | Best RMSE | Strategy | Features | Improvement |
|---------|---------------|-----------|----------|----------|-------------|
| Poland Long | 4,580.89 | 4,541.05 | base_context | 10 | +0.87% |
| Poland Short | 27,367.70 | 22,541.28 | base_context | 37 | +17.64% |
| UAE Contracts | 465,037.04 | 360,266.87 | base_context | 88 | +22.53% |
| UAE Transactions | 32,489,660 | 26,353,228 | base_context | 31 | +18.89% |

## Key Findings

1. **Feature selection is essential**  using all 300-700 context features hurts performance
2. **10-100 features optimal**  diminishing returns beyond this
3. **ase_context wins**  ranked #1 on all 4 datasets
4. **Context alone insufficient**  context_only strategies underperform baseline

## Data Sources

- results/rental_poland_long_search_20260119_132926/
- results/rental_poland_short_search_20260119_133601/
- results/rental_uae_contracts_search_20260119_134236/
- results/sales_uae_transactions_search_20260119_135741/
`

---

## 3. Paper Enhancements

### 3.1 Features in Code NOT in Paper

| Feature | Code Location | Paper Addition |
|---------|---------------|----------------|
| **Fold Variance Features** | _aggregate_fold_statistics() | Add 3.3 subsection |
| **Global Statistics** | include_global_stats=True | Add to 3.2 |
| **Interaction Features** | include_interactions=True | Add to 3.2 |
| **Feature Cleanup Pipeline** | CleanupConfig | Not needed (engineering detail) |
| **Extended Quantiles** | Q05, Q10, Q20, Q33, Q66, Q80, Q90, Q95 | Mention in 3.1 |

---

### 3.2 Suggested Paper Additions

#### A. Add Algorithm 1 Pseudocode (HIGH PRIORITY)

The paper references "Algorithm 1" but doesn't show it. Add to Section 3:

`latex
\begin{algorithm}[t]
\caption{SCE Construction with Cross-Fitting}
\begin{algorithmic}[1]
\REQUIRE Dataset $\mathcal{D} = \{(x_t, y_t)\}_{t=1}^n$, levels $, folds $
\ENSURE Enriched dataset with context features
\STATE Partition indices into $ folds: , \ldots, I_M$
\FOR{each fold  = 1, \ldots, M$}
    \STATE $\mathcal{D}_{-m} \gets \{(x_s, y_s) : s \notin I_m\}$ \COMMENT{Out-of-fold data}
    \FOR{each level  = 1, \ldots, K$}
        \FOR{each  \in I_m$}
            \STATE $\mathcal{N}_k(t) \gets \{s \in \mathcal{D}_{-m} : g_k(x_s) = g_k(x_t)\}$
            \STATE $\phi^{(k)}(x_t) \gets \mathcal{S}_k(\{y_s : s \in \mathcal{N}_k(t)\})$
        \ENDFOR
    \ENDFOR
\ENDFOR
\STATE $\Phi(x_t) \gets [\phi^{(1)}(x_t), \ldots, \phi^{(K)}(x_t)]$ for all $
\RETURN $\{(x_t, \Phi(x_t), y_t)\}_{t=1}^n$
\end{algorithmic}
\end{algorithm}
`

---

#### B. Add Fold Variance Section (MEDIUM PRIORITY)

Add to Section 3 after Eq. 4:

`latex
\subsection{Uncertainty Quantification via Fold Variance}

Cross-fitting naturally provides uncertainty estimates for context statistics. 
For each statistic $, we compute its variance across folds:
\begin{equation}
\sigma^2_{\text{fold}}(s) = \frac{1}{M-1} \sum_{m=1}^M (s_m - \bar{s})^2
\label{eq:fold_variance}
\end{equation}
where $ is the statistic computed on fold $ and $\bar{s}$ is the mean.
This provides downstream models with reliability information for each context feature.
We optionally include fold standard deviation, lower bound ($\bar{s} - 2\sigma$), 
and upper bound ($\bar{s} + 2\sigma$) as additional features.
`

---

#### C. Add Global Statistics Definition (LOW PRIORITY)

Add to Section 3.2:

`latex
At the coarsest level (=0$), we define the \emph{global} neighborhood as the 
entire dataset: $\mathcal{N}_0(t) = \{1, \ldots, n\}$. The global context 
$\phi^{(0)}(x_t)$ captures dataset-wide distributional properties and provides 
a universal baseline for hierarchical backoff.
`

---

#### D. Add Interaction Features Definition (LOW PRIORITY)

Add to Section 3.2:

`latex
Beyond single-column hierarchies, we compute \emph{interaction features} 
for categorical column pairs. For columns $ and $, the interaction 
neighborhood is:
\begin{equation}
\mathcal{N}_{ij}(t) = \{s : c_i(x_s) = c_i(x_t) \land c_j(x_s) = c_j(x_t)\}
\end{equation}
This captures fine-grained segment effects (e.g., `apartments in Dubai Marina'').
`

---

#### E. Add Feature Selection Discussion (HIGH PRIORITY)

The code shows feature selection is critical. Add to Section 5 or Results:

`latex
\paragraph{Feature Selection.}
Ablation studies reveal that using all context features (300--700 depending on 
hierarchy depth) degrades performance compared to selecting 30--100 features 
by tree importance. This suggests that many context statistics are redundant 
or noisy, and that parsimonious feature selection is essential for optimal 
SCE performance.
`

---

## 4. Files to Delete

| File | Reason |
|------|--------|
| results/HONEST_RESULTS_SUMMARY.md | Outdated, refers to old leaky configuration |
| results/experiment_results.json | Single old experiment, superseded by search results |

---

## 5. Summary Checklist

### Skills Updates
- [x] Update sce-algorithm/SKILL.md results section
- [x] Update sce-algorithm/references/CODE_MAP.md verified results
- [x] Expand sce-experiments/SKILL.md with search documentation
- [x] Fix paper-publication/references/FIGURE_SPECS.md naming

### New References
- [x] Create sce-algorithm/references/AUDIT_2026-01-19.md
- [x] Create sce-experiments/references/SEARCH_STRATEGIES.md
- [x] Create sce-experiments/references/RESULTS_2026-01-19.md

### Paper Updates
- [ ] Add Algorithm 1 pseudocode
- [ ] Add fold variance equation (Eq. 5)
- [ ] Add feature selection discussion
- [ ] (Optional) Add global/interaction definitions

### Cleanup
- [x] Delete results/HONEST_RESULTS_SUMMARY.md
- [x] Delete results/experiment_results.json

---

*Generated by Auditor Agent, 2026-01-19*
*Updated by Tech Lead, 2026-01-19*
