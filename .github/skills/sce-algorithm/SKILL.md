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
| Evaluation protocol | scripts/run.py (`_run_sce_enrichment`, `_split_dataset`) |
| Leakage diagnostics | scripts/diagnostics/ |
| Guard tests | tests/test_leakage_guards.py, tests/test_no_pre_split_fit.py |

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

## Evaluation Protocol (BINDING — do not regress)

This section supersedes all guidance dated 2026-01-16 that previously lived
here. The remediation plan `docs/plan/2026-04-18_leakage_safe_remediation_plan.md`
is the source of truth.

1. **Split FIRST, then enrich.** The sequence is: Load → Sample → **Split** →
   fit SCE on train only → transform test with train-derived statistics.
   Enriching before the split lets test-set targets flow into features and
   inflates results. (Earlier versions of this document claimed the opposite —
   that guidance was wrong and produced the inflated pre-2026-04 numbers.)
2. **No target-derived features in evaluation.** Ratio features
   (`y / group_mean`) and any feature computed from the row's own target are
   prohibited in evaluation runs. `include_relative_features` stays off.
3. **Temporal data requires temporal cross-fit.** `split.strategy = "temporal"`
   with random KFold cross-fitting is a hard error (`scripts/run.py` guard).
   Use `cross_fit_strategy = "time"` or `"rolling"` with `time_col` set.
4. **Train-only preprocessing.** Categorical encoding, pruning, and cleanup are
   fit on the training partition only and applied to test.
5. **Search selection never touches the test set.** `FeatureCombinationSearch`
   scores candidates on an internal validation split (`val_fraction`,
   `val_strategy="tail"` for temporal data); only the selected winners are
   refit on full train and evaluated once on test (`eval_set="test"`).
6. **Run provenance.** Every run writes `metadata.json` (git SHA, config hash,
   seed, `run_grade`). Paper numbers come only from `run_grade=report-grade`
   runs.

## Leakage Checklist

- Are any aggregates computed from `target`? (must be cross-fitted, OOF)
- Are relative features using `target`? (prohibited in evaluation)
- Are folds disjoint and applied before aggregation?
- Does the temporal variant ever use future rows for statistics? (must not)
- Is any model/feature selection decision based on test metrics? (must not)

## Diagnostics (run before trusting any result)

| Diagnostic | Tool | Pass criterion |
|---|---|---|
| Permuted target | `scripts/diagnostics/permuted_target.py` | SCE improvement ≈ 0 on shuffled targets |
| Shuffled groups | `scripts/diagnostics/shuffled_groups.py` | Context features lose predictive power |
| Cross-fit A/B | `scripts/diagnostics/crossfit_ab.py` | OOF vs naive gap quantified and reported |
| Feature dominance | `scripts/diagnostics/feature_dominance.py` | No single context feature dominates importance |

## Common Patterns

### Group stats computation
- Group-by on categorical keys (single columns + interactions)
- Aggregate the target with mean/median/std/quantiles/count
- Join back as prefixed features (`{level}__{stat}`)

### Hierarchical backoff
- Small groups receive statistics from coarser levels (sce/stats.py,
  `apply_hierarchical_backoff`), global stats as final fallback.

### Cross-fitting (Equation 4)
- `StatisticalContextEngine._fit_transform_cross_fitted`: row t's features come
  from folds that exclude t. Temporal variants use `TimeSeriesSplit` so
  statistics never come from the future; earliest rows may stay unenriched
  (NaN → dropped downstream — report the count).

## Historical Note (results prior to 2026-04)

Results generated before the leakage-safe remediation (including the
"+56% RMSE reduction" figure and the 2026-01-19 search table that previously
appeared here) used enrich-before-split and test-set-based selection. They are
**not citable**. Regenerate all numbers with report-grade runs under the
current protocol before using them anywhere.

## Resources

- [Equations reference](references/EQUATIONS.md)
- [Code mapping](references/CODE_MAP.md)
- [Remediation plan](../../../docs/plan/2026-04-18_leakage_safe_remediation_plan.md)
- Historical audits (pre-remediation, do not follow their conclusions):
  references/AUDIT_2026-01-15.md, references/AUDIT_2026-01-19.md

## Gotchas

- Small groups create unstable statistics; minimum count + backoff handle this.
- Level names use double underscores (`__`) as delimiters because column names
  may contain single underscores (`city__room_type` → `["city", "room_type"]`).
- Tree-based models exploit SCE features far better than linear models.
- `sce/search.py` is EXPERIMENTAL (low coverage); treat its outputs as
  exploratory unless the run is report-grade.
