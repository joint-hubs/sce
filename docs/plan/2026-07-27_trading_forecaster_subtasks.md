# Trading Forecaster — Subtask Decomposition

**Source PRD:** `docs/plan/2026-07-27_trading_forecaster_prd.md`
**Epic slug:** `equity-forecast-pipeline`
**Total subtasks:** 32 (across 8 slices)
**Lock-in date:** 2026-07-27 (Mateusz GATE 1 defaults)

## Legend
- **Type:** `feat` | `fix` | `chore` | `test` | `docs` | `refactor` | `spike`
- **Estimate (t-shirt):** `S` ≤ ½ day · `M` 1–2 days · `L` 3–5 days · `XL` > 1 week → re-split
- **Slice dependency graph:** `S1 ← S2 ← S3 ← S4 ← S5 ← S6 ← S7` ; `S8 ← {S4, S6}` (parallel to S7)
- **AC form:** Given / When / Then (1–3 per subtask, concrete + testable)
- **Locked decisions baked in:** Q1 S&P 500 · Q2 `ret_1d` SCE target · Q3 free RSS first · Q4 interaction allow-list → upstream `sce/` w/ ADR, `transform_partial` → `equity/sce/` wrapper · Q5 horizons `{1,5,10,21,63}` · Q6 5y/63d/63d walk-forward · Q7 quantile heads ship now · class renamed `FinBERTSententer` → `FinBERTScorer` with `SentimentScorer.classify(text) -> dict` interface.

---

## S1 — Data acquisition + canonical schema
*Value: a leakage-safe, reproducible equity dataset any downstream model can consume.*

### S1.1 — Equity universe + delisting-aware registry
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S1 · **BlockedBy:** —

**AC:**
- **Given** the locked universe is S&P 500 (Q1) and the registry lives under `configs/equity/`,
- **When** a user loads `EquityDataLoader(universe="sp500", start=..., end=...)`,
- **Then** the loader resolves a TOML config under `configs/equity/sp500.toml` whose `universe_file` includes delisted tickers with their delist dates, and the loader exposes a `universe()` accessor returning `(ticker, listed_at, delisted_at)` tuples.

**DoD:** TOML config committed; loader raises on missing/delisted tickers; unit test asserts `delisted_at IS NOT NULL` for ≥10 known delistees from the reference list.

### S1.2 — OHLCV ingestion + parquet stores
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S1 · **BlockedBy:** —

**AC:**
- **Given** an instrument universe and a date range,
- **When** `EquityDataLoader.fetch_prices()` is invoked,
- **Then** it produces `prices.parquet` partitioned by `period_close_ts` year/month with columns `ticker, period_close_ts, open, high, low, close, adj_close, volume, vwap`, and `period_close_ts` is timezone-aware exchange-local close (16:00 ET for US).

**DoD:** Parquet schema validated by `pandera`; integration test pulls a 30-day yfinance slice and a Kaggle historical slice and asserts primary-key uniqueness on `(ticker, period_close_ts)`.

### S1.3 — Point-in-time text join + `published_at_guard` diagnostic
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S1 · **BlockedBy:** S1.2

**AC:**
- **Given** a `prices` table and a raw `articles` table with `(ticker, published_at, text, source)`,
- **When** the join layer assigns articles to periods,
- **Then** an article is bound to period `P` iff `period_close(P-1) < published_at <= period_close(P)`, holidays roll to the next trading period via `pandas_market_calendars`, and `published_at_guard` (in `equity/diagnostics/`) asserts `published_at <= period_close_ts` for every joined row.

**DoD:** Diagnostic CLI exits non-zero on a synthetic injected violation; docstring + README section under `equity/data/README.md` describe the join semantics.

---

## S2 — FinBERT sentiment + per-period aggregation
*Value: sentiment features with provable point-in-time alignment, reusable across any forecasting model.*

### S2.1 — `FinBERTScorer` + `SentimentScorer` interface
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S2 · **BlockedBy:** S1.3

**AC:**
- **Given** the PRD's typo `FinBERTSententer`/`sententer` is corrected to `FinBERTScorer`,
- **When** a user constructs `FinBERTScorer(model_name="ProsusAI/finbert", batch_size=...)`,
- **Then** it implements `SentimentScorer.classify(text: str) -> dict` returning `{pos, neg, neu, score=pos-neg}` and the class lives at `equity/sentiment/finbert.py`.

**DoD:** Interface defined as a `Protocol` in `equity/sentiment/base.py`; unit test stubs a classifier and asserts the protocol contract.

### S2.2 — Idempotent per-article score cache
- **Type:** `feat` · **Estimate:** `S` · **Slice:** S2 · **BlockedBy:** S2.1

**AC:**
- **Given** a corpus of `n` articles,
- **When** `FinBERTScorer.score_corpus(articles)` is run twice with identical inputs,
- **Then** the second run returns the same `sentiment_score` for every article and performs zero model forward passes (verified by call-counting the HF pipeline).

**DoD:** Cache key = sha256 of `(text, model_name)`; cache stored under `.cache/sentiment/` (gitignored); test asserts re-run latency < 5% of cold run on a 1k-article corpus.

### S2.3 — Time-decayed count-weighted per-(ticker, period) aggregation
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S2 · **BlockedBy:** S2.2

**AC:**
- **Given** per-article scores and the `sentiment_halflife_days` config (default 5),
- **When** the per-period aggregator runs over a `(ticker, period)` partition,
- **Then** it emits `sentiment_score_P = sum(w_t * score_t) / sum(w_t)` with `w_t = exp(-(period_close_P - published_at_t) / halflife)`, alongside `sentiment_pos/neg/neu` aggregates and `n_articles`, and writes one row per `(ticker, period)`.

**DoD:** Unresolvable articles drop to a separate `market_wide_sentiment` aggregate; numeric test on a hand-built fixture asserts the formula reproduces expected values within 1e-9.

### S2.4 — VADER fallback scorer
- **Type:** `feat` · **Estimate:** `S` · **Slice:** S2 · **BlockedBy:** S2.1

**AC:**
- **Given** the FinBERT pipeline is too slow for the paper-trading latency budget,
- **When** a user instantiates `VADERScorer()` (also implementing `SentimentScorer.classify`),
- **Then** it returns `{pos, neg, neu, score}` from `vaderSentiment` and a classifier-swap env var (`SENTIMENT_SCORER=vader|finbert`) flips the default at paper-trader boot.

**DoD:** `equity/sentiment/vader.py` ships with one fixture-based test; metadata records the active scorer.

---

## S3 — Lag-aware technical + sentiment feature layer
*Value: a strictly past-only feature block, usable independently of SCE and the forecaster.*

### S3.1 — Technical indicators module
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S3 · **BlockedBy:** S1.2

**AC:**
- **Given** the OHLCV table from S1 and `FeatureConfig(indicators=[...])`,
- **When** `equity/features/technical.py` builds features for period `t`,
- **Then** it produces log returns (1/5/10/21d), SMA/EMA (5/10/21/63), RSI(14), MACD(12,26,9), realised vol (21/63d), volume Z-score (21d), ATR(14), Bollinger bands — every indicator using only `prices[:t]` (i.e. inputs exclude `close[t]`).

**DoD:** Fixture test with hand-checked RSI/MACD values; column list exported via `FeatureConfig.describe()`.

### S3.2 — Lag layer (`shift`, `rolling_mean`, `rolling_std`)
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S3 · **BlockedBy:** S3.1, S2.3

**AC:**
- **Given** the indicator + sentiment columns,
- **When** `LagConfig(windows=[1,3,5,10,21], methods=["shift","rolling_mean","rolling_std"])` is applied,
- **Then** each base feature spawns `{feature}_lag{N}`, `{feature}_rollmean{N}`, `{feature}_rollstd{N}` columns, all using only rows at or before `t - N`.

**DoD:** `equity/features/lag.py` shipped; numeric test asserts `_lag1` row `t` equals the unsourced row at `t-1`.

### S3.3 — `lookahead_indicator` diagnostic guard
- **Type:** `test` · **Estimate:** `S` · **Slice:** S3 · **BlockedBy:** S3.1

**AC:**
- **Given** the engineered feature matrix,
- **When** `lookahead_indicator` recomputes each indicator from `prices[:t]` only,
- **Then** it asserts equality with the stored feature and the CLI exits non-zero on a synthetic injected violation (e.g. SMA window including the current row).

**DoD:** Lives under `equity/diagnostics/`; README documents the leak it catches.

---

## S4 — SCE equity enrichment (rolling cross-fit)
*Value: SCE's leakage-safe hierarchical context, proven on a financial regime — the core research contribution.*

### S4.1 — `equity/sce/enrich.py` wrapper + rolling cross-fit config
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S4 · **BlockedBy:** S3.2

**AC:**
- **Given** the feature matrix from S3 and `ret_1d` as the SCE target (Q2 lock-in),
- **When** `EquityContextEnricher.fit_transform(features, time_col="period_close_ts")` runs,
- **Then** it wraps `sce.StatisticalContextEngine` with `ContextConfig(use_cross_fitting=True, cross_fit_strategy="rolling", n_folds=5, min_group_size=20)` and emits `{column}_ret_1d_{stat}` context features for the curated aggregation set.

**DoD:** Smoke test on a 500-row fixture passes; integration test on a 1y S&P 500 slice runs < 5 min.

### S4.2 — Equity hierarchy columns + curated interaction config
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S4 · **BlockedBy:** S4.1

**AC:**
- **Given** the hierarchy DAG `market | sector | industry | mktcap_bucket | ticker` plus `time_bucket`,
- **When** `EquityHierarchyConfig(...)` is built,
- **Then** it passes `categorical_cols` = `[time_bucket, sector, industry, mktcap_bucket, ticker]` and the curated interaction allow-list (`sector×mktcap_bucket`, `ticker×time_bucket`, `sector×time_bucket`) to SCE, never enabling `include_interactions=True` blindly.

**DoD:** Cardinality test asserts generated group count ≤ `|ticker| × |time_bucket|` (not the all-pairs product).

### S4.3 — SPIKE/ADR — upstream interaction allow-list in `sce/`
- **Type:** `spike` · **Estimate:** `M` · **Slice:** S4 · **BlockedBy:** S4.1

**AC:**
- **Given** the equity pipeline needs an explicit `interactions=[(...)]` allow-list on `ContextConfig`,
- **When** the spike ADR (`docs/adr/NNN-sce-interaction-allowlist.md`) is filed,
- **Then** it proposes the change as a backward-compatible optional list (boolean remains the default), names the consumer (= equity pipeline S4.2), evaluates 2 alternatives (manual interaction columns in `equity/sce/` vs upstream change), and lists consequences for other SCE users.

**DoD:** ADR committed under `docs/adr/`; spike issue closed with decision (`upstream accepted` / `wrapper rejected`); one PR-link or local patch summary in the ADR footer.

### S4.4 — `transform_partial` wrapper in `equity/sce/` (not upstream)
- **Type:** `feat` · **Estimate:** `S` · **Slice:** S4 · **BlockedBy:** S4.1

**AC:**
- **Given** Q4 locks `transform_partial` to `equity/sce/` (not upstream),
- **When** `EquityContextEnricher.transform_partial(new_rows, refit_boundary_ts)` is called,
- **Then** it re-`fit`s the engine on `(train_start, refit_boundary_ts]` and returns `transform(new_rows)` statistics, mirroring the upstream semantics without modifying `sce/`.

**DoD:** Lives at `equity/sce/transform_partial.py`; unit test asserts output shape matches a fresh `fit`+`transform` on the same window.

### S4.5 — `walk_forward_monotonicity` + `survivorship_check` diagnostics
- **Type:** `test` · **Estimate:** `S` · **Slice:** S4 · **BlockedBy:** S4.1

**AC:**
- **Given** a walk-forward fold definition (S6.1 geometry),
- **When** `walk_forward_monotonicity` is invoked,
- **Then** it asserts `train_max_ts < val_min_ts` and `val_max_ts < test_min_ts` per fold, exits non-zero on violation, and `survivorship_check` asserts the historical universe contains ≥ N delisted tickers from a reference list.

**DoD:** Both diagnostics registered in the `equity.diagnostics` entry-point group; one passing run on a 5y S&P 500 slice captured.

### S4.6 — Reuse SCE's `permuted_target` / `shuffled_groups` / `crossfit_ab` / feature-dominance on `ret_1d`
- **Type:** `test` · **Estimate:** `M` · **Slice:** S4 · **BlockedBy:** S4.1

**AC:**
- **Given** the enriched equity dataset with `ret_1d` as target,
- **When** the SCE diagnostic suite (`scripts/diagnostics/`) is invoked via the equity runner,
- **Then** `permuted_target` shows SCE advantage < 1%, `shuffled_groups` shows no advantage with shuffled `sector`/`ticker`, `crossfit_ab` reports a non-positive `leakage_signal_pp`, and feature-dominance top-3 share < 70%.

**DoD:** Diagnostic JSON outputs committed to `docs/figures/equity/` (placeholder path); README explains how to re-run.

---

## S5 — Multi-horizon two-layer forecaster (offline train + predict)
*Value: a buildable, leakage-safe forecaster producing `pred_hN` for all configured horizons.*

### S5.1 — Sector-head model per horizon (H = {1, 5, 10, 21, 63})
- **Type:** `feat` · **Estimate:** `L` · **Slice:** S5 · **BlockedBy:** S4.6

**AC:**
- **Given** the enriched dataset and `HorizonConfig(horizons=[1,5,10,21,63])` (Q5),
- **When** `SectorHeadForecaster.fit()` runs,
- **Then** it trains one XGBoost regressor per horizon (5 models) on its own `ret_hN` label, producing out-of-fold `pred_sector_hN` for every row using the same rolling cross-fit folds as SCE enrichment (no in-sample predictions leak into the residual layer).

**DoD:** OOF prediction coverage = 100% of train rows; per-horizon model artefact persisted with horizon in the filename.

### S5.2 — Instrument-residual model with OOF residual labels
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S5 · **BlockedBy:** S5.1

**AC:**
- **Given** the OOF `pred_sector_hN` from S5.1,
- **When** `InstrumentResidualForecaster.fit()` runs,
- **Then** it computes `resid_hN = ret_hN - pred_sector_hN` (OOF only, no in-sample leakage), trains per-horizon residual XGBoost models whose inputs are the S5.1 feature block plus `pred_sector_hN`, and the final forecast is `pred_hN = pred_sector_hN + pred_resid_hN`.

**DoD:** Numeric test asserts `pred_hN - ret_hN` MSE on a held-out fold ≤ baseline (no-residual) MSE; ridge regression kept as a weak reference.

### S5.3 — `forward_target_isolation` diagnostic
- **Type:** `test` · **Estimate:** `S` · **Slice:** S5 · **BlockedBy:** S5.1

**AC:**
- **Given** the feature matrix handed to either forecaster layer,
- **When** `forward_target_isolation` runs,
- **Then** it asserts no `ret_h{N}` column for `N ∈ {1,5,10,21,63}` appears in the feature matrix, and `ret_hN` is excluded from SCE's `categorical_cols` and aggregation targets.

**DoD:** CLI exits non-zero on an injected leakage; one negative test (intentional leak) confirms detection.

### S5.4 — Quantile heads (LightGBM quantile regression)
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S5 · **BlockedBy:** S5.1

**AC:**
- **Given** Q7 locks quantile heads in for the S5 slice,
- **When** `QuantileHeadForecaster.fit()` runs,
- **Then** it adds a LightGBM quantile model per horizon with `quantiles=[0.05, 0.5, 0.95]` (alpha=0.05 → 90% interval), trained jointly with the XGBoost point head on the same OOF folds, and emits `pred_hN_q05`, `pred_hN_q50`, `pred_hN_q95` per row.

**DoD:** Coverage test on a held-out fold asserts empirical coverage of the 90% interval is in `[0.85, 0.95]`; artefact persisted per horizon.

### S5.5 — Single-fold train/test smoke run on the historical corpus
- **Type:** `test` · **Estimate:** `S` · **Slice:** S5 · **BlockedBy:** S5.2, S5.4

**AC:**
- **Given** the full historical corpus (S1) enriched (S4) with point + quantile heads (S5.1, S5.4),
- **When** the single-fold runner (`equity.forecaster.run_smoke`) executes,
- **Then** it produces one `predictions.parquet` per horizon covering the test slice and a `metadata.json` with `git_sha, config_hash, seed, run_grade="exploratory"`.

**DoD:** Smoke run completes in CI; artefact paths declared; README documents how to promote to `diagnostic` / `report-grade`.

---

## S6 — Walk-forward backtest + metrics
*Value: the report-grade numbers that prove (or refute) SCE's value on equities, fully reproducible.*

### S6.1 — Walk-forward runner (5y / 63d / 63d, roll by 63d)
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S6 · **BlockedBy:** S5.5

**AC:**
- **Given** Q6 locks the geometry to 5y train / 63d val / 63d test, rolling by 63d,
- **When** `WalkForwardRunner.run()` executes,
- **Then** it produces `~N_years/0.25` folds where per fold it fits on `[t-W_train, t-W_val)`, validates on `[t-W_val, t)`, tests on `[t, t+W_test)`, re-fits SCE inside the train window per fold, and reuses `walk_forward_monotonicity` (S4.5) to assert monotonicity.

**DoD:** No shuffling anywhere; SEED fixed per fold; integration test runs on a 3y fixture and asserts expected fold count.

### S6.2 — Per-horizon RMSE / MAE / directional hit-rate
- **Type:** `feat` · **Estimate:** `S` · **Slice:** S6 · **BlockedBy:** S6.1

**AC:**
- **Given** the per-fold predictions,
- **When** `equity/metrics/accuracy.py` runs,
- **Then** it reports RMSE, MAE, and `sign(pred)==sign(realised)` hit-rate per horizon, aggregated across folds with mean and std.

**DoD:** Output JSON schema validated; one fixture-based test asserts hand-computed metrics match within 1e-6.

### S6.3 — Decile long/short Sharpe/Sortino + horizon selection
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S6 · **BlockedBy:** S6.2

**AC:**
- **Given** per-fold predictions across all `H` horizons,
- **When** `equity/metrics/sharpe.py` runs,
- **Then** it builds a daily-decile long/short portfolio (top/bottom decile, equal-weighted within, dollar-neutral) per horizon, annualises PnL by `sqrt(252)`, computes Sharpe and Sortino, and selects `hN*` per regime by validation-period Sharpe (never on test).

**DoD:** Horizon choice recorded in `metadata.json`; test asserts the selection rule never reads from the test slice.

### S6.4 — Baseline-vs-SCE comparison report + run metadata
- **Type:** `docs` · **Estimate:** `M` · **Slice:** S6 · **BlockedBy:** S6.3

**AC:**
- **Given** the baseline (no SCE features) and SCE runs share model class, hyper-params, and folds,
- **When** `equity/reports/compare.py` runs,
- **Then** it produces a side-by-side Markdown/CSV table of ΔRMSE / Δhit-rate / ΔSharpe per horizon, plus `metadata.json` containing `git_sha, config_hash, seed, run_grade, chosen_horizon, sce_diagnostics_summary`.

**DoD:** Report committed under `docs/reports/equity/`; metadata schema validated; README links to the report.

---

## S7 — Paper-trading loop
*Value: a no-capital live validation harness, ready to hand off to the future trading-bot epic.*

### S7.1 — Live OHLCV stream ingestion (yfinance → daily close)
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S7 · **BlockedBy:** S6.4

**AC:**
- **Given** the universe and the live data API,
- **When** `equity/paper_trader/stream.py` runs in `loop` mode,
- **Then** it polls the OHLCV feed at close-1min, builds a `prices` micro-batch with the S1 canonical schema, and emits an idempotent Kafka-like local queue file per trading day.

**DoD:** Holiday/half-day handling via `pandas_market_calendars`; one dry-run day captured end-to-end.

### S7.2 — Free RSS news feed ingestion (Q3 lock-in)
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S7 · **BlockedBy:** S7.1

**AC:**
- **Given** Q3 locks free RSS as the first paper-trading news source,
- **When** the RSS poller runs each minute,
- **Then** it parses configured feeds into `(ticker, published_at, text, source)` rows matching the S1 article schema, writes them to `articles.parquet`, and exposes a `NEWS_SOURCES` env var to add feeds without code change.

**DoD:** One end-to-end dry run on a 1h RSS window; ticker resolver (`yfinance` search + manual override file) documented in README.

### S7.3 — Weekly SCE refit cadence + online transform pipeline
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S7 · **BlockedBy:** S7.2, S4.4

**AC:**
- **Given** the rolling window ends at the previous trading close,
- **When** the scheduler triggers a refit (default weekly, configurable),
- **Then** it calls `EquityContextEnricher.transform_partial(new_rows, refit_boundary_ts)` (S4.4) and the forecaster consumes the resulting context features without blocking the live loop for > N minutes (configurable budget).

**DoD:** Refit cadence + budget surfaced as metrics; one simulated 5-day run captures a full refit cycle.

### S7.4 — Paper position sheets + append-only ledger + mark-to-market
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S7 · **BlockedBy:** S7.3

**AC:**
- **Given** the live predictions,
- **When** `equity/paper_trader/loop.py` ticks at each close,
- **Then** it emits a paper position sheet (long/short deciles), marks-to-market the previous sheet, and appends `{ts, ticker, action, pred_hN, price, paper_pnl}` to an append-only ledger file (NDJSON, immutable on disk).

**DoD:** Ledger append-only invariant enforced by file permissions / write-once helper; one 5-day paper run captured.

---

## S8 — Diagnostics & report-grade gate for equity
*Value: the credibility layer — no equity number is citable until it passes the same gate SCE enforces on its existing benchmarks.*

### S8.1 — Consolidated equity diagnostics gate
- **Type:** `feat` · **Estimate:** `M` · **Slice:** S8 · **BlockedBy:** S4.6, S6.4

**AC:**
- **Given** the equity-specific diagnostics from S4.5 + S5.3 plus the SCE-reused diagnostics from S4.6,
- **When** `equity.diagnostics.run_gate` is invoked on a backtest run,
- **Then** it executes all checks (S4.5, S4.6, S5.3, S6.1's monotonicity, S4's interaction cardinality), aggregates pass/fail into a single JSON, and exits non-zero on any failure.

**DoD:** Single CLI entrypoint; CI integration example in README; gate exit code 0 on the S6.4 report-grade run.

### S8.2 — Report-grade promotion wired into run metadata
- **Type:** `feat` · **Estimate:** `S` · **Slice:** S8 · **BlockedBy:** S8.1

**AC:**
- **Given** a run that passes the gate (S8.1),
- **When** `promote_to_report_grade(run_id)` is called,
- **Then** it writes `run_grade="report-grade"` into the run's `metadata.json` (immutable copy on object storage / read-only ledger), mirroring SCE's protocol.

**DoD:** Downgrade path documented; promotion audit-log appended; one demo promotion captured.

### S8.3 — Paper figures (mirroring `docs/figures/paper/`)
- **Type:** `docs` · **Estimate:** `M` · **Slice:** S8 · **BlockedBy:** S8.2

**AC:**
- **Given** the report-grade run,
- **When** `equity/reports/figures.py` runs,
- **Then** it produces figures parallel to SCE's paper set: M1 RMSE-improvement (baseline vs SCE), M2 feature-contributions, M3 horizon/strategy ranking, plus per-horizon tables in CSV/PDF/PNG, written under `docs/figures/equity/`.

**DoD:** Figure generation is idempotent (skip if up-to-date); one rendered set committed for the report-grade run; README links to the figures.