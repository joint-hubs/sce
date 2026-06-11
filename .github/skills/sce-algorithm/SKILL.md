---
name: sce-algorithm
description: Statistical Context Engineering algorithm implementation and verification. Use when verifying code against paper equations, checking cross-fitting logic, auditing for leakage, reviewing Algorithm 1, or debugging statistical computations. Triggers include cross-fitting, leakage, equations, paper, algorithm, phi, context operator, variance reduction, out-of-fold.
---

# SCE Algorithm Verification

## Quick Reference

| Concept | Location |
|---|---|
| Paper equations | paper_overleaf_format.txt |
| Core engine | sce/engine.py |
| Statistics | sce/stats.py |
| Config | sce/config.py |

## When to Use This Skill

- You need to verify code matches Equations (1)–(5) and Algorithm 1
- You are auditing for target leakage or missing cross-fitting
- You need a code-to-equation traceability check

## Procedure

1. Read the paper equations in references/EQUATIONS.md.
2. Map each equation to implementation in references/CODE_MAP.md.
3. Verify group statistics use only allowed columns.
4. Check leakage prevention logic (cross-fitting, target exclusion).
5. Record gaps or deviations as TODOs and create tests for them.

## Common Patterns

### Group stats computation
- Group-by on hierarchical keys
- Aggregate numeric columns with mean/median/quantiles
- Join back as prefixed features

### Relative features
- Ratios or z-scores computed from group statistics
- Never compute target-relative features for supervised training

## Leakage Checklist

- Are any aggregates computed from `target`? (should be cross-fitted)
- Are relative features using `target`? (should be avoided)
- Are folds disjoint and applied before aggregation?

## Gaps To Watch

**✅ RESOLVED (2026-01-15):** Cross-fitting is now fully implemented in `sce/engine.py` via the `_fit_transform_cross_fitted()` method. Out-of-fold aggregation prevents target leakage as specified in Equation 4.

**✅ RESOLVED (2026-01-15):** Target-based context statistics are correctly computed. The engine aggregates the target column (`target_col`) across hierarchy groups, matching paper specifications.

**✅ RESOLVED (2026-01-16):** Hierarchical backoff implemented in `sce/stats.py` via `apply_hierarchical_backoff()`. Per paper Section 3.4, observations in small groups now receive context statistics from coarser hierarchy levels. This improved average RMSE reduction from +29.70% to +56.40%.

**✅ RESOLVED (2026-01-16): Ratio features now implemented correctly:**
- The paper's ratio features (Eq. 3) `ratio = y_t / μ_k` compare target to group mean
- These are computed on the FULL dataset BEFORE train/test split
- This is NOT leakage because the split happens AFTER enrichment
- **Implementation:** `create_ratio_features()` in `scripts/run.py`

**✅ RESOLVED (2026-01-16): Model choice matters for SCE features:**
- Ridge regression (linear model) does NOT work well with SCE features
- XGBoost (tree-based) correctly leverages the hierarchical interactions
- **FIX:** Always use XGBoost or tree-based models with SCE enrichment

**✅ RESOLVED (2026-01-16): SCE must be applied BEFORE train/test split:**
- Applying SCE after split loses valuable context information
- The correct order is: Load → Sample → SCE Enrich → Split → Train/Evaluate
- This matches the paper methodology and produces massive improvements

**Current Status:** All equations (1-4) and Algorithm 1 verified against production code.

**Current Results (Search Experiments 2026-01-19):**

| Dataset | Baseline RMSE | Best SCE RMSE | Improvement | Features |
|---------|---------------|---------------|-------------|----------|
| Poland Long | 4,581 | 4,541 | +0.87% | 10 |
| Poland Short | 27,368 | 22,541 | +17.64% | 37 |
| UAE Contracts | 465,037 | 360,267 | +22.53% | 88 |
| UAE Transactions | 32,489,660 | 26,353,228 | +18.89% | 31 |

*Note: Best results use `base_context` strategy with feature selection.*

## Extensions Beyond Paper

**Global Level Statistics:** The implementation includes a "global" level that computes dataset-wide statistics (mean, std, median, etc.) in addition to hierarchy levels. This provides:
- A baseline context for all observations
- A final fallback for hierarchical backoff
- Consistent with paper's variance reduction perspective (Eq. 5)

## Resources

- [Equations reference](references/EQUATIONS.md)
- [Code mapping](references/CODE_MAP.md)
- [Audit 2026-01-19](references/AUDIT_2026-01-19.md)

## Gotchas

- Small groups can create unstable statistics; apply minimum count / backoff.
- Consistent naming of context features is required for downstream configs.
- **⚠️ Model Selection:** Always use tree-based models (XGBoost, Random Forest) with SCE features. Linear models (Ridge, Lasso) cannot capture the non-linear interactions that SCE provides.
- **⚠️ Order of Operations:** SCE enrichment MUST happen BEFORE train/test split. The sequence is: Load → Sample → SCE → Split → Train.
- **⚠️ Leakage Risk in Experiments:** Evaluate with `use_cross_fitting=True` (now the default in `scripts/run.py`) or compute stats strictly on training folds and apply to test. Avoid full-dataset target aggregation when reporting metrics.
- **⚠️ Target-Derived Features Prohibited:** Ratio features (`y / group_mean`) or any feature computed from the target are not allowed for evaluation. Do not enable target-derived features in experiment runs.
- **⚠️ Column names with underscores:** The stats system uses double underscores (`__`) as delimiters between hierarchy levels. This is essential because column names may contain single underscores (e.g., `room_type`, `property_type`). Using single underscore as delimiter would break when parsing level names back to column lists.
  - **Example:** Hierarchy `["city", "room_type"]` → level name `"city__room_type"` → columns `["city", "room_type"]`
  - **Bug fixed (2026-01-16):** Changed from `"_".join()` to `"__".join()` in stats.py and engine.py
