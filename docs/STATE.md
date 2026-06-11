# Project State

> Living document. Update at the end of every working session.
> Last update: **2026-06-11** (evening session)

## Where we are

- **Version:** 0.3.5 on PyPI (`stat-context`), `import sce`.
- **Paper context:** ICML rebuttal plans from 2026-03-29 in `docs/plan/`.
- **Leakage-safe remediation plan** (`docs/plan/2026-04-18_leakage_safe_remediation_plan.md`):
  **DONE** — all P0/P1/P2 tasks implemented and committed (2026-06-11).
  Includes: temporal cross-fit guard, train-only encoding/pruning, run metadata
  (`git SHA + config hash + seed + run_grade`), diagnostics suite
  (`scripts/diagnostics/`: permuted target, shuffled groups, cross-fit A/B,
  feature dominance).
- **Dataset expansion:** M5, Rossmann, Walmart, Melbourne configs + prepare
  scripts committed. Parquets are **not** in git (Kaggle redistribution rules) —
  rebuild locally with `scripts/prepare_new_datasets.py` / `scripts/prepare_m5_dataset.py`.
- **Search selection bias FIXED (2026-06-11):** `FeatureCombinationSearch` now
  selects candidates on an internal validation split (`val_strategy="tail"` for
  temporal data); winners are refit on full train and evaluated once on test
  (`eval_set` column distinguishes them). Pruning trace also moved off test.
  **All search results produced before this fix are inflated — not citable.**
- **Report-grade gate wired (2026-06-11):** `run.py` loads the latest
  diagnostics from `results/diagnostics/<dataset>/` and computes feature
  dominance in-run; promotion to report-grade works end-to-end.
- **Tests:** full suite green locally (116+ passed, Python 3.13, `.venv`).
- **Working tree:** clean as of 2026-06-11; main is ahead of `origin/main`
  (push pending — see next steps).

## Report-grade status (2026-06-11, post search-fix, git f6c427a0)

| Dataset | Diagnostics | Report-grade run | RMSE improvement (test, unbiased) |
|---|---|---|---|
| rental_poland_short | full, all pass | **PROMOTED** | **+10.97%** |
| m5_store_dept_daily | full, all pass | **PROMOTED** | +1.14% |
| melbourne_housing | full, all pass | **PROMOTED** | +2.19% |
| rental_poland_long | permuted+shuffled **FAIL** | BLOCKED (correct) | +1.22% real ≈ noise (permuted gives +3.1%) |
| walmart / rossmann / UAE ×2 | 20k-subsample diagnostics queued | pending | — |

Honest takeaway: poland_long's SCE advantage is within noise at n≈1000 — the
gate caught it. Promotion also now requires every diagnostic to report
`pass=true` (`diagnostic_failed:*` blocks).

## What is NOT done (next steps, in order)

Driven by `docs/plan/2026-04-18_release_1_0_plan.md`:

1. **Push main to origin** and confirm CI is green (first CI run with the new
   test files; CI matrix is Python 3.9–3.12, locally tested on 3.13 only).
2. **R0-1 remainder:** rerun all datasets with fresh metadata; at least one run
   with `run_grade=report-grade`.
3. **R0-2 ⚠ DECYZJA:** license — CC-BY-NC-4.0 blocks a 1.0 library release.
   Suggested: Apache-2.0 for code, paper stays CC-BY-NC. Mateusz must decide.
4. **R0-3:** heavy deps (xgboost/lightgbm/catboost) → optional extras.
   Note: lightgbm+catboost were just ADDED to core deps for the experiment work —
   this must be reversed for 1.0.
5. **R0-4:** remove deprecated API (`hierarchy`, `include_quantiles`, …).
6. **R0-5/6/7:** public API freeze, remove `print()` from library, repo cleanup.
7. Then R1 (quality gates) → R2 (docs) → R3 (TestPyPI rc1 → 1.0.0).

## How to run

```powershell
.venv\Scripts\python.exe -m pytest -q          # tests
python scripts/run.py --dataset rental_poland_short   # single experiment
python scripts/run.py --all --search --report  # full paper workflow
```

Kaggle credentials live in `configs/kaggle.json` (gitignored — never commit).

## Gotchas

- `pip install stat-context` but `import sce` — known naming inconsistency.
- `configs/rolling_window_report_defaults.toml` is shared defaults, not a
  dataset config (`list_datasets()` skips TOMLs without `dataset.path`).
- `data/raw/` (~370 MB) and derived Kaggle parquets are gitignored.
