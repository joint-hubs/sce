# Project State

> Living document. Update at the end of every working session.
> Last update: **2026-06-12** (overnight run results)

## Where we are

- **Version:** **0.4.0 RELEASED to PyPI** (2026-06-12, tag `v0.4.0`); `pip install stat-context`, `import sce`. CI green on 3.9–3.12; clean-venv install smoke-tested.
- **License:** code relicensed to **Apache-2.0** (2026-06-12); paper stays CC-BY-NC.
- **Active benchmark set (5):** rental_poland_short, melbourne_housing,
  m5_store_dept_daily, walmart_weekly, rossmann_daily — all report-grade
  PROMOTED. Blocked configs moved to `configs/experimental/` (poland_long,
  sales_uae, rental_uae) pending full-data reruns.
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

## Report-grade status (2026-06-12, post overnight run, git 2925a9ea)

| Dataset | Diagnostics (full data) | Report-grade run | RMSE improvement (test, unbiased) |
|---|---|---|---|
| rental_poland_short | full, all pass | **PROMOTED** | **+10.97%** |
| m5_store_dept_daily | full, all pass | **PROMOTED** | +1.14% |
| melbourne_housing | full, all pass | **PROMOTED** | +2.19% |
| walmart_weekly | full (420k), permuted −0.24% ✅, shuffled −0.90% ✅ | **PROMOTED** | **+6.35%** |
| rossmann_daily | full (844k), permuted −0.32% ✅, shuffled −0.40% ✅ | **PROMOTED** | **+9.90%** |
| rental_poland_long | permuted+shuffled **FAIL** | BLOCKED (correct) | +1.22% real ≈ noise (permuted gives +3.1%) |
| sales_uae_transactions | 20k subsample: **permuted FAIL (+24.5%!)** | blocked — investigate | +8.7% real, NOT trustworthy |
| rental_uae_contracts | 20k subsample: SCE **hurts** (−6.7% real), shuffled FAIL | blocked — investigate | — |

**crossfit_ab anomaly (FYI, not blocking):** `leakage_signal_pp` is negative for both
Walmart (−9.0%) and Rossmann (−3.6%), meaning the no-CF model slightly outperforms
the CF model on the test set. Expected for large datasets where each row's
contribution to its group mean is negligible (OOF adds noise without removing
meaningful leakage). Permuted-target and shuffled-groups diagnostics confirm zero
leakage — this is not a red flag. `crossfit_ab` currently has no `pass` field
(informational only); gate uses permuted+shuffled as blocking criteria.

UAE caveat: 20k random rows out of 1M/5.5M leave most groups nearly empty; UAE
numbers are subsample artifacts. `sales_uae` permuted-target failure (+24.5%)
is a serious red flag regardless. `rental_uae` SCE hurts — both need investigation.

Honest takeaway: poland_long's SCE advantage is within noise at n≈1000 — the
gate caught it correctly. 5 out of 8 datasets now promoted to report-grade.

## What is NOT done (next steps, in order)

Driven by `docs/plan/2026-04-18_release_1_0_plan.md`:

1. ~~Push + CI~~ DONE (2026-06-12): main pushed, CI green on 3.9–3.12
   (fixed: ruff format across repo, lazy requests/tqdm import in data/download.py).
2. ~~R0-1~~ DONE: all 5 active datasets have report-grade promoted runs.
3. ~~R0-2~~ DONE (2026-06-12): code relicensed to Apache-2.0.
4. ~~R0-3~~ DONE (2026-06-12): lightgbm+catboost → `[models]` extra;
   sklearn floor raised to 1.4 (README quickstart uses root_mean_squared_error).
   xgboost stays core for now (full extraction is a 1.0 task).
5. ~~Release 0.4.0~~ DONE (2026-06-12): tag `v0.4.0` → TestPyPI → PyPI →
   GitHub Release, all jobs green; fresh-venv install verified.
6. **R0-4:** remove deprecated API (`hierarchy`, `include_quantiles`, …).
7. **R0-5/6/7:** public API freeze, remove `print()` from library, repo cleanup.
8. Then R1 (quality gates) → R2 (docs) → R3 (TestPyPI rc1 → 1.0.0).
9. **UAE datasets:** full-data diagnostics reruns, then re-promote configs from
   `configs/experimental/` if they pass. **Night-run script READY (not launched):**
   `%TEMP%\sce_night_run_uae.cmd` → log `results/night_run_uae.log`.
   RAM caveat: rental_uae is 4.1 GB in memory; watch the first minutes.

## Paper figures status (2026-06-12)

- **REGENERATED from report-grade runs:** `results_consolidated` (B1),
  `feature_contributions` (B4), `summary_table.tex/.txt` — committed under
  `docs/figures/paper/`. Source: `run.py --all --run-grade report-grade`
  (5 datasets, avg RMSE improvement +6.11%).
- **DELETED (stale, old protocol, not citable):** `summary_fig1–6` — they were
  built from January `categorical_mode_batch_summary_*` and March search runs.
- **PENDING fresh inputs:** cross-model figures (fig1–6) need
  `--compare-categorical-modes` runs, appendix figures need `--search` artifacts.
- **Pre-remediation result dirs archived** to `results/archive_pre_remediation/`
  (47 dirs from March/April: categorical compares, batch summaries, old search)
  so `--latest` aggregation only sees post-remediation runs.

## Full batch RUNNING on GCP (launched 2026-06-12 ~17:45)

- **VM:** `sce-night-run`, c2d-standard-16 **spot** (16 vCPU, 64 GB),
  europe-central2-b, project `dochubs`, max-run-duration 12 h (auto-STOP).
  Cost ≈ $0.20/h spot → ~$1.5–2.5 for the whole batch.
- **Startup script:** `scripts/gcp_night_startup.sh` (committed) — clones main,
  installs `.[models,viz]`, pulls parquets + diagnostics from
  `gs://sce-night-dochubs`, runs the full sequence below, uploads
  `done/night_results.tar.gz` + `DONE` marker, powers off.
- **Partial sync:** results rsync to `gs://sce-night-dochubs/partial/` every
  5 min, so a spot preemption loses at most one step. If preempted (VM state
  TERMINATED before DONE marker): `gcloud compute instances start sce-night-run
  --zone=europe-central2-b` — the startup script is idempotent (fresh clone).
- **Local watcher:** detached PowerShell (`%TEMP%\sce_gcp_watcher.ps1`) polls
  the bucket every 5 min for up to 16 h; on DONE it downloads and unpacks into
  `results_gcp_<ts>/` in the repo. Log: `results/gcp_watcher.log`.
- **Check status:** `gcloud compute instances list` (RUNNING/TERMINATED) and
  `gcloud storage ls gs://sce-night-dochubs/done/`.

Local fallback script also exists: `%TEMP%\sce_night_run_full.cmd`. Sequence
(same on both):
1. Cross-model compare: 7 models × 5 datasets (fast GBDTs first, slow sklearn last)
2. Search × 5 datasets (validation-selected protocol)
3. UAE full-data diagnostics (sales 1M, rental 5.5M — RAM-heavy, isolated)
4. `generate_categorical_mode_batch_summary --latest` (fresh runs only)
5. All three figure scripts
6. `scripts/night_report.py` → `results/night_report_<ts>.md` (full markdown
   report: experiments, search, compares, diagnostics, figure freshness)

Requires a clean git tree at launch (report-grade gate). The older partial
scripts (`sce_night_run_uae.cmd`, `sce_night_run_paper.cmd`) are superseded.

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
