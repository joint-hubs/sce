# Project State

> Living document. Update at the end of every working session.
> Last update: **2026-07-27** (FOC-48 slice S1 — equity data-acquisition package, on review branch)

## Equity forecasting pipeline (FOC-47/48) — branch `foc-48-s1-data-acquisition`

**Status (2026-07-27):** Slice S1 (Data acquisition + canonical schema) IMPLEMENTED
on feature branch `foc-48-s1-data-acquisition` (NOT merged to main; in Linear
review). 3 commits: `61c39db` (S1.1), `fa9351b` (S1.2), `70824f2` (S1.3).

New sibling package `equity/` (independent of `sce/`, mirrors `sce/io` registry
pattern; NOT imported into `sce/`):
- `equity/data/loader.py` — `EquityDataLoader(universe, start, end, period="1d",
  tickers=None)`; `.universe()` (delisting-aware `(ticker, listed_at,
  delisted_at)` tuples); `.fetch_prices()` (yfinance OHLCV → Hive-partitioned
  `prices.parquet`, cols `ticker,period_close_ts,open,high,low,close,adj_close,
  volume,vwap`); `.fetch_articles()` (seed → `articles.parquet`);
  `.join_articles_to_prices()` (point-in-time join via XNYS calendar; rule
  `period_close(P-1) < published_at <= period_close(P)` + holiday roll-forward).
- `equity/data/schema.py` — pandera `prices_schema` (tz-aware
  `America/New_York`) + `articles_schema` (tz-aware UTC) + `validate_*` +
  PK-uniqueness helpers. **First pandera use + first tz-aware code in repo.**
- `equity/data/fetch.py` — `fetch_yfinance_ohlcv` (in-process, no subprocess;
  VWAP proxy `(h+l+c)/3`) + `fetch_articles_from_seed`.
- `equity/data/registry.py` — `UniverseInfo` + `list_universes()`/
  `get_universe_info()` (globs `configs/equity/*.toml`).
- `equity/diagnostics/published_at_guard.py` — CLI
  `python -m equity.diagnostics.published_at_guard`; **exits non-zero on
  PIT-join violation** (first non-zero-exit diagnostic in repo).
- Configs: `configs/equity/sp500.toml` (`[universe]`/`[prices]`/`[articles]`);
  `sp500_universe.csv` seed (20 current + 13 verified delisted S&P 500 tickers
  with public-record delist dates); `articles_seed.csv`.
- `pyproject.toml`: new `[equity]` extra (yfinance, pandas_market_calendars,
  pandera); `packages.find` extended to `["sce*", "equity*"]`; `all` aggregator
  extended; `equity.__version__ = 0.3.0`.

**Tests:** `tests/equity/` — 37 passed + 2 skipped (live yfinance + Kaggle
integration tests gated by `SCE_EQUITY_LIVE_TEST=1`; no network on default run).
Regression on `sce` (io + models): green.

**Decisions locked (GATE 1, PRD `docs/plan/2026-07-27_trading_forecaster_prd.md`):**
Q1 S&P 500 · Q2 `ret_1d` SCE target · Q3 free RSS first · Q4 interaction
allow-list (ADR in S4) · Q5 horizons `{1,5,10,21,63}` · Q6 5y/63d/63d
walk-forward · Q7 quantile heads.

**Assumptions to confirm in review:** (1) reference delisted-tickers = committed
seed CSV (full historical-constituents Kaggle dataset deferred to S1.2 follow-up,
seed swappable via TOML `universe_file`); (2) `published_at` canonical tz = UTC;
(3) Cam Nugent Kaggle slug `camnugent/sandp500`; (4) ticker-alias mapping (e.g.
`ENRN`→`ENE`) not implemented — S1.3 filters out-of-window tickers; (5) `equity/`
sibling package does not touch `stat-context` core deps or `import sce`.

**Next (after review/merge):** S2 (FinBERT sentiment + per-period aggregation)
→ S3 (lag-aware features) → S4 (SCE equity enrichment). Subtask plan:
`docs/plan/2026-07-27_trading_forecaster_subtasks.md`.

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
| sales_uae_transactions | **full 996k: permuted +0.47% ✅ (clean), SCE real −1.70%** | BLOCKED — no benefit | subsample +8.7% was an artifact; SCE does not help on full data |
| rental_uae_contracts | **full-data OOM on 64 GB (exit 137)** | rerun pending (high-mem VM) | unknown until rerun |

### Full-data UAE verdict (2026-06-13, from GCP batch)

The 20k subsamples were misleading on BOTH UAE datasets:

- **sales_uae_transactions** — on full 995,975 rows the permuted-target advantage is
  **+0.47% (PASS)**, not the +24.5% the subsample showed. So there is **no leakage
  red flag**. But SCE's real advantage is **−1.70%** — it makes RMSE slightly worse
  than baseline. shuffled_groups `pass=False` is *correct*: with real_advantage ≤ 0
  there is nothing to validate. Honest conclusion: **SCE simply does not help this
  dataset** (high-cardinality transaction IDs, little group structure). Stays in
  `configs/experimental/`, reason updated from "leakage" → "no benefit on full data".
- **rental_uae_contracts** — 5.48M rows (4.3 GB raw) OOM-killed all three diagnostics
  on the 64 GB batch VM (exit 137 right after "Total hierarchy levels created: 7").
  Full-data verdict still unknown.

  **Rerun #1 (FAILED, 2026-06-13):** VM `sce-rental-uae-run` c2d-highmem-32 launched
  but `/usr/bin/time` was not installed on Ubuntu 22.04 — `run_step()` used it as
  a wrapper, so Python never ran (exit=127). Start and end timestamps were identical;
  memory was untouched (249 GB free throughout). Fix: added `time` to the
  `apt-get install` line in `scripts/gcp_rental_uae_startup.sh`.

  **Rerun #2 (RUNNING, 2026-06-13 ~13:47 UTC):** VM `sce-rental-uae-run`
  **c2d-highmem-16 (16 vCPU, 128 GB, spot)** (quota limit prevented c2d-highmem-32).
  128 GB = 2× headroom vs the 64 GB that OOM'd. Runs only the 3 rental_uae
  diagnostics via `scripts/gcp_rental_uae_startup.sh`, uploads to separate
  `gs://sce-night-dochubs/done_rental_uae/` marker. Local watcher restarted:
  `%TEMP%\sce_rental_uae_watcher.ps1` (polls every 5 min for 6 h → unpacks into
  `results_gcp_rental_uae_<ts>/`; log `results/rental_uae_watcher.log`).
  Check: `gcloud compute instances list --filter=name=sce-rental-uae-run` and
  `gcloud storage ls gs://sce-night-dochubs/done_rental_uae/`.

### NaN / linear-model fixes (2026-06-13)

The batch had 4 hard failures (exit 1): `ridge` and `gradient_boosting` on
`rental_poland_short` + `melbourne_housing`. Root cause: those sklearn estimators
reject NaN inputs (SCE feature blocks contain NaN for small/empty groups), unlike
the GBDT libraries which handle NaN natively. Fix in `sce/models.py`:
- non-GBDT sklearn models (ridge, random_forest, extra_trees, gradient_boosting) are
  now wrapped in an imputing `Pipeline` (`inf→nan` sanitize → median `SimpleImputer`
  with `keep_empty_features=True` → estimator); ridge additionally gets
  `StandardScaler` and uses `RidgeCV` (auto-alpha over a wide grid) instead of fixed
  `alpha=1`, which prevented coefficient blow-ups.
- `extract_feature_importance` unwraps the pipeline's final estimator.
- Verified: all 4 previously-crashing runs now finish (exit 0); GBDT results
  unchanged (xgboost rental_poland_short +6.73%); full suite 123/123 green.

**Known limitation (not a bug):** `ridge` still posts catastrophic RMSE on
heavy-tailed targets (e.g. rental_poland_short target spans 436 → 1,010,368, median
3,344). Linear regression cannot fit such tails; tree models clip naturally. This is
an inherent linear-baseline property and was already present in the batch
(walmart ridge −246%). Ridge stays in the cross-model comparison as a weak baseline.

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

## Full batch on GCP — COMPLETED (2026-06-12 20:14 UTC, ~4.5 h, ~$1.5 spot)

Results in `results_gcp_20260613_1052/`, unpacked and copied into the repo
(`docs/figures/paper` + `docs/figures/appendix` updated, gitignored run dirs into
`results/`). **Not committed yet.** Outcome of the 6 phases:
1. Cross-model compare: **31/35 OK** (4 NaN crashes — now fixed, see above).
   `categorical_mode_batch_summary` + summary_fig1–6 regenerated from 31 runs.
2. Search ×5: **5/5 OK** (validation-selected protocol).
3. UAE full-data diagnostics: sales_uae **3/3 OK** (verdict above);
   rental_uae **0/3 OOM** (rerun pending).
4. Aggregate summary: OK. 5. Figures: summary + appendix OK;
   `generate_figures.py` failed (needs `experiment_results.json` from a `--all`
   run, which the batch did not produce — harmless, the paper figures come from
   the other two scripts). 6. night_report: OK
   (`results/night_report_20260612_201406.md`).

## Earlier: full batch launch notes (2026-06-12 ~17:45)

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
