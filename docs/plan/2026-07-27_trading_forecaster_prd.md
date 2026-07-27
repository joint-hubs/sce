# Equity Forecasting Pipeline — SCE-Enriched, Sentiment-Aware, Multi-Horizon

**Created:** 2026-07-27
**Epic slug:** `equity-forecast-pipeline`
**Owner:** SPEC (planning) — Mateusz Stachowicz (sign-off)
**Status:** Draft for GATE 1 review
**Builds on:** SCE `v0.4.0` (PyPI `stat-context`, `import sce`), `StatisticalContextEngine`, `ContextConfig`, rolling cross-fit, `scripts/diagnostics/`
**Out-of-scope explicit:** broker integration, order execution, real capital. A separate later epic (`equity-trading-bot`) will consume this pipeline's forecasts.

---

## 1. Context & motivation

SCE today is a **leakage-safe hierarchical target-statistic enrichment** for tabular regression, validated on five report-grade datasets (rental, Rossmann, Walmart, Melbourne, M5) under a strict diagnostics gate (permuted-target, shuffled-groups, crossfit_ab, feature-dominance). Its identity is *leakage-safety*, and its temporal mode (`cross_fit_strategy="rolling"`, `train_max < val_min` via `time_col`) is exactly the guarantee financial-ML needs.

Financial time-series forecasting is the canonical setting where leakage destroys credibility: future news bleeds into a period, indicators peek at future closes, walking-forward is replaced by a shuffled K-fold, survivorship bias creeps in via currently-listed tickers. We want to (a) prove SCE generalises to a noisy, non-stationary, text-augmented financial regime, and (b) build a real, buildable forecasting pipeline whose features and validation are leakage-safe by construction. The pipeline will eventually feed a trading bot, but **this epic stops at paper trading** (simulated positions on a live/streaming feed, no real orders, no real capital).

## 2. Goals / Non-goals

### Goals
- G1. A reproducible pipeline that ingests equity OHLCV + finance text, aligns them point-in-time, and emits one canonical row per `(instrument, period)`.
- G2. FinBERT-based per-instrument-per-period sentiment scores with provable `published_at <= period_close`.
- G3. A lag-aware feature layer (technical indicators + sentiment, raw + lagged over configurable windows) — strictly past-only.
- G4. SCE enrichment over an explicit equity hierarchy (instrument × sector × industry × market-cap bucket × time bucket) using **rolling temporal cross-fit**.
- G5. A multi-horizon, multi-layer forecaster (sector-head + instrument-residual, per-horizon heads) for forward returns.
- G6. A walk-forward backtest (no look-ahead, no shuffle) reporting RMSE/MAE per horizon, directional hit-rate, and Sharpe/Sortino of a simple long/short rule; followed by paper trading on a live feed.
- G7. A finance-specific leakage diagnostics suite that reuses SCE's existing diagnostics and adds equity-specific checks.

### Non-goals (this epic)
- Broker integration, order routing, real execution, real capital.
- Live trading, portfolio optimisation, position sizing beyond the simple long/short rule used for the Sharpe metric.
- Alternative assets (crypto, FX, futures, options). Equities only.
- High-frequency / intraday-tick forecasting. The smallest period is one trading day.
- A production-grade serving system. The pipeline runs as batch + a paper-trading loop.

## 3. Locked scope decisions (do not re-litigate)

- **Asset class: equities (stocks).**
- **Epic scope: pipeline only.** Broker/execution = separate later epic.
- **Validation depth: walk-forward backtest + paper trading** (simulated positions, NO real capital, NO real orders).
- **Multi-horizon:** several forecast horizons supported; the best horizon is chosen empirically (a parameter, not a fixed choice).
- **Multi-layer / hierarchical model structure is required** ("models should have their own hierarchy too"). Concrete design in §7.
- **Period granularity: daily** is the base case; the architecture must allow weekly/monthly resampling without re-architecting. Intraday is out.
- **Primary model backend: gradient-boosted trees** (XGBoost core, LightGBM/CatBoost via `[models]` extra), consistent with SCE's existing model factory in `sce/models.py`. A linear baseline (`RidgeCV`) is kept as a weak reference, mirroring SCE's cross-model comparison.
- **Sentiment model: FinBERT (`ProsusAI/finbert`)** as the default; the interface must allow swapping to a lighter/faster classifier for paper-trading latency.

## 4. Data

### 4.1 Sources
- **OHLCV:** `yfinance` for live/delayed prices (free, sufficient for daily equities). A Kaggle dataset (e.g. "S&P 500 stock data" or "NYSE/NASDAQ fundamentals") provides the historical training corpus and survivorship-bias-controlled universe.
- **Text:** Kaggle finance-news datasets (e.g. "Financial News Dataset" / `Kaggle` financial news, plus a Twitter/Reddit finance stream for the live paper-trading phase). For paper trading, a live RSS / news API is the streaming source.
- **Reference / instrument metadata:** sector, industry, market-cap bucket, listing/delisting dates (point-in-time, see §9.4). Sourced from a Kaggle fundamentals dataset or a static snapshot. The static snapshot is acceptable for the backtest **if** delisting-aware universe construction is documented.

### 4.2 Canonical row: one `(instrument, period)` per row
- Primary key: `(ticker, period_close_ts)`. `period_close_ts` is the timezone-aware timestamp of the period's last trading second (e.g. 16:00 ET for daily US equities).
- Columns at the raw price layer: `ticker, period_close_ts, open, high, low, close, adj_close, volume, vwap`.
- Forward targets (added at feature time, never exposed as features): `ret_h1, ret_h5, ret_h10, ret_h21` — forward N-period log returns, where `hN` is the N-trading-day forward return. The set of horizons is config-driven.
- Sentiment layer (one or more rows per `(ticker, period)`): `ticker, period_close_ts, text_id, published_at, sentiment_pos, sentiment_neg, sentiment_neu, sentiment_score, n_articles, source`.
- SCE/context layer (added during enrichment): `{column}_{target}_{stat}` features per SCE's naming convention.

### 4.3 Text→period alignment (point-in-time correctness)
- An article is **assigned to period `P`** iff `period_close(P-1) < published_at <= period_close(P)`. I.e. only news *known before* period `P`'s close can influence period `P`'s row. This is the same past-only rule SCE's rolling cross-fit enforces for target statistics; here it is enforced at the join layer.
- An article is **eligible for period `P`'s features** (those that feed the model predicting `ret_h{N}` from period `P`'s close) iff `published_at <= period_close(P)`.
- Multiple articles per `(ticker, period)` are aggregated to the period level by **time-decayed, count-weighted** sentiment: `sentiment_score_P = sum(w_t * score_t) / sum(w_t)` with `w_t = exp(-(period_close_P - published_at_t)/halflife)`. `halflife` is config. Counts (`n_articles`) become their own features.
- Articles that cannot be ticker-resolved are dropped from the per-instrument layer; an aggregate "market-wide sentiment" feature is computed at the period level (no ticker) and reused as a global stat in SCE.

### 4.4 Storage
- Parquet on local disk (gitignored), partitioned by `period_close_ts` year/month. Live phase adds an appendable store (parquet/duckdb). No data redistribution; Kaggle rebuilds happen locally, mirroring the existing SCE dataset prepare scripts (`scripts/prepare_new_datasets.py`).

## 5. Architecture / pipeline (6 stages)

```
[1 Acquire] -> [2 Sentiment] -> [3 Features (lagged)] -> [4 SCE enrich] -> [5 Forecast] -> [6 Validate]
                                                                              |
                                                  (live stream) -> [4' enrich-online] -> [5' forecast-online] -> [6' paper trade]
```

### Stage 1 — Data acquisition
- Module: `equity/data/loader.py` with `EquityDataLoader(universe, start, end, period="1d")`.
- Output: two raw parquet stores — `prices.parquet` and `articles.parquet` (with `published_at`, `ticker`, `text`, `source`).
- Reuses SCE's dataset-registry pattern (`sce/io/`): each universe/source pair is registered with a TOML config under `configs/equity/`.
- Universe construction is **delisting-aware**: includes delisted tickers in the historical universe (so a model trained on today's S&P 500 doesn't see survivorship). Documented per dataset.

### Stage 2 — Sentiment
- Module: `equity/sentiment/finbert.py` with `FinBERTSententer(model_name="ProsusAI/finbert", batch_size=...)`.
- Loads FinBERT via HuggingFace `transformers`; caches scores to disk keyed by `text_id` hash (idempotent re-runs).
- Output: per-article `{pos, neg, neu}` probabilities → single `sentiment_score = pos - neg` plus the three probabilities. Caches per `(ticker, period)` aggregates from §4.3.
- Lighter classifier hook: `sententer` is a small interface (`classify(text) -> dict`); a trivial VADER-based fallback is provided for the live phase where FinBERT latency is too high.
- Tests assert `published_at <= period_close_ts` for every joined row.

### Stage 3 — Feature engineering (lag-aware)
- Module: `equity/features/technical.py` and `equity/features/lag.py`.
- Technical indicators: log returns (1d, 5d, 10d, 21d), SMA/EMA (windows 5/10/21/63), RSI(14), MACD(12,26,9), realised volatility (21d, 63d), volume Z-score (21d), ATR(14), Bollinger bands.
- **All indicators use only data strictly before `period_close_ts`.** A guard test asserts that for any feature column `f` and row at time `t`, `f(t)` equals `f(t)` recomputed from `prices[:t]` alone (i.e. indicator inputs do not include `close[t]`).
- Lag layer: every sentiment and indicator feature is lagged over windows `L = {1, 3, 5, 10, 21}` (config: `LagConfig(windows=[1,3,5,10,21], methods=["shift","rolling_mean","rolling_std"])`). Naming: `{feature}_lag{N}`, `{feature}_rollmean{N}`, `{feature}_rollstd{N}`.
- Sentiment lags include `sentiment_score_lag1`, `_lag3`, … and rolling means/stds — captures sentiment drift and momentum.
- Config: `FeatureConfig(indicators=[...], lag_windows=[...], sentiment_halflife_days=5)`.

### Stage 4 — SCE enrichment
- Module: `equity/sce/enrich.py` wrapping `sce.StatisticalContextEngine`.
- Hierarchy columns (passed as `categorical_cols`):
  - `ticker` (instrument)
  - `sector` (e.g. GICS sector)
  - `industry` (sub-industry)
  - `mktcap_bucket` (quantile bucket: micro / small / mid / large / mega)
  - `time_bucket` (e.g. calendar month or quarter — captures regime; only past time buckets are seen via rolling cross-fit)
  - **Interactions** enabled: `include_interactions=True` produces e.g. `sector×mktcap_bucket`, `ticker×time_bucket` (selected interactions are an explicit config to control cardinality).
- Target for SCE: the **lagged forward return** is NOT used as the SCE target (that would leak the future). The SCE target is a **realised past return** proxy that exists at feature time: e.g. `ret_1d` of the period just closed (the same-period realised return). This makes the group statistics "the recent return context this stock's peer group lives in" — past-only by construction. The forward targets `ret_hN` are kept separate and used only as model labels.
- Cross-fit: `use_cross_fitting=True`, `cross_fit_strategy="rolling"`, `time_col="period_close_ts"`, `n_folds=5` (config). This reuses SCE's temporal guard directly: `train_max < val_min`.
- `min_group_size` raised relative to SCE defaults (e.g. 20) — financial groups are noisy; small groups back off to global stats, the SCE-builtin behaviour.
- Aggregations: `MEAN, MEDIAN, STD, Q05, Q20, Q80, Q95, COUNT` (the SCE default set), applied to the realised-return target.
- Online/paper-trading path: `engine.transform(new_rows)` is called per batch of new rows; the engine holds the fitted state from the rolling training window. Re-fit cadence is configurable (e.g. weekly).

### Stage 5 — Forecasting (multi-horizon, multi-layer)
See §7 for the concrete design.

### Stage 6 — Validation
See §8.

## 6. SCE integration (hierarchy design, cross-fit, diagnostics, extensions)

### 6.1 Hierarchy
The equity hierarchy is a DAG of grouping columns, in increasing granularity:

`market | sector | industry | mktcap_bucket | ticker` and the time dimension `time_bucket`.

SCE's `categorical_cols` + `include_interactions=True` already supports cross-column groupings. We use a curated set:
- Global: `time_bucket`
- Market-regime: `time_bucket × sector`
- Sector context: `sector`, `sector × mktcap_bucket`
- Industry context: `industry`
- Instrument context: `ticker`, `ticker × time_bucket` (captures recent per-stock regime)

High-cardinality interactions (e.g. `ticker × time_bucket` with monthly buckets) are gated by `min_group_size` and the back-off to global stats.

### 6.2 Rolling cross-fit (reuse)
- `ContextConfig(target_col="ret_1d", categorical_cols=[...], use_cross_fitting=True, cross_fit_strategy="rolling", time_col="period_close_ts", n_folds=5, min_group_size=20, include_interactions=True)`.
- The existing temporal guard (`train_max < val_min`) is exactly the financial "no look-ahead" guarantee; we rely on it and add tests that assert the monotonic-fold invariant on the equity data.
- Online (paper-trading) transform uses the latest fitted fold's full-train statistics — the same `transform` path SCE uses for the test set.

### 6.3 Diagnostics reused
Run unchanged from `scripts/diagnostics/`:
- `permuted_target` — permute `ret_1d` (the SCE target). Pass: SCE advantage < 1% on shuffled target. A leaking financial encoder would "improve" even on noise.
- `shuffled_groups` — shuffle `sector`/`ticker` labels. Pass: no SCE advantage with shuffled categories.
- `crossfit_ab` — informational; expect negative `leakage_signal_pp` on large equity panels (as seen on Walmart/Rossmann in SCE's own STATE).
- feature-dominance — top-3 share < 70%.

### 6.4 Finance-specific diagnostics (new, this epic)
- `lookahead_indicator` — for each indicator feature, recompute from `prices[:t]` only and assert equality with the stored feature.
- `published_at_guard` — assert `published_at <= period_close_ts` for every sentiment-joined row.
- `forward_target_isolation` — assert no `ret_h{N}` column appears in the feature matrix passed to the model; assert `ret_h{N}` are not in `categorical_cols` or aggregation targets.
- `walk_forward_monotonicity` — for each walk-forward fold, assert `train_max_ts < val_min_ts` and `val_max_ts < test_min_ts`.
- `survivorship_check` — assert delisted tickers are present in the historical universe (count check vs a delisting reference list).

### 6.5 Engine extensions (proposed, minimal)
Two small extensions to `sce/` are proposed in this epic (ADR-eligible):
1. **Explicit interaction allow-list** — `ContextConfig(interactions=[("sector","mktcap_bucket"), ("ticker","time_bucket"), ...])` instead of `include_interactions=True` (all pairs). Today `include_interactions` is boolean; with high-cardinality equity columns, the all-pairs product is wasteful. Proposed as a backward-compatible optional list (boolean remains the default for existing users).
2. **Online transform with rolling refit hook** — a `transform_partial(new_rows, refit_boundary_ts)` method that uses statistics fit on `(train_start, refit_boundary_ts]`. Needed for the paper-trading loop's weekly refit cadence. Implementable today by calling `transform(new_rows)` after a `fit` on the desired window; the extension is just a convenience that avoids refit-bookkeeping in user code.

If either extension is contentious, the spec-review loop or a Spike ADR will resolve. If rejected, the pipeline works around them by (1) computing interactions manually and (2) re-`fit`-ing the engine on the rolling window.

## 7. Modeling: multi-horizon + multi-layer hierarchy (concrete design)

### 7.1 Targets
- Forward log returns at horizons `H = {1, 5, 10, 21}` trading days: `ret_h1, ret_h5, ret_h10, ret_h21`. Config: `HorizonConfig(horizons=[1,5,10,21])`.
- Best horizon is chosen empirically by walk-forward validation per (instrument universe, regime) — selected as the horizon with the best Sharpe of the long/short rule (§8.3). The chosen horizon is recorded in run metadata, mirroring SCE's `metadata.json` traceability.

### 7.2 Two-layer model: sector-head + instrument-residual

**Layer 1 — Sector-head model** (one model per sector, or one model with sector embeddings; we pick **one model with sector as a categorical feature + SCE sector-context features**, simpler and matches SCE's existing single-model design):
- Input: lagged technical + sentiment features + SCE context features (`sector_*`, `industry_*`, `mktcap_bucket_*`, `time_bucket_*`, `ticker_*`).
- Output: per-horizon heads — a multi-output regressor (one XGBoost per horizon, or a single multi-output LGBM with `num_horizons` heads). Concrete choice: **one XGBoost regressor per horizon** (`H` models), each trained on its own `ret_hN` label. This keeps per-horizon hyper-parameter tuning simple and matches SCE's existing one-target model factory.
- Predicts `pred_sector_hN(ticker, t)` — the sector-level expected return for the period.

**Layer 2 — Instrument-residual model** (residualises the sector prediction to the instrument):
- Residual label: `resid_hN = ret_hN - pred_sector_hN` (computed out-of-fold only — the sector-head's predictions on its own training set would leak; we use sector-head OOF predictions as the residual target, exactly analogous to SCE's OOF cross-fitting).
- Input: the same feature block **plus** `pred_sector_hN` as a feature (the sector signal as context).
- Output: `pred_resid_hN(ticker, t)`.
- Final forecast: `pred_hN = pred_sector_hN + pred_resid_hN`.

This is the "models have their own hierarchy too" requirement: a sector-level signal is composed with an instrument-level residual, both under rolling cross-fitting, both leakage-safe by construction (OOF residuals).

### 7.3 Per-horizon heads and horizon selection
- Each horizon `hN` has its own (sector-head, residual) pair. Horizons are independent models — no horizon mixing in a single model.
- Horizon selection is a **post-validation** choice, reported with the metrics that selected it. The pipeline ships all `H` horizons; downstream consumers (the future trading-bot epic) pick the horizon by their own decision rule.

### 7.4 Model factory
Reuses `sce/models.py` (`XGBRegressor` core, LightGBM/CatBoost via `[models]`, `RidgeCV` as a weak baseline). Hyper-parameters are fixed per horizon for the report-grade run (mirroring SCE's protocol of "identical hyperparameters for baseline vs SCE").

### 7.5 Baseline vs SCE comparison
- **Baseline**: lagged technical + sentiment features only (no SCE context features).
- **SCE**: baseline + SCE context features.
- Same model class, same hyper-parameters, same walk-forward folds. SCE's value-add is the delta in RMSE/MAE/hit-rate/Sharpe. This mirrors SCE's existing benchmark protocol exactly.

## 8. Validation protocol

### 8.1 Walk-forward backtest (no look-ahead, no shuffle)
- Folds: rolling-window walk-forward. Concrete: train window `W_train = 5y`, validation `W_val = 63d` (used only for early-stopping / horizon selection), test `W_test = 63d`, then roll forward by `W_test`. This yields ~`N_years/0.25` test folds for a `N_years` history.
- Per fold: fit on `[t-W_train, t-W_val)`, validate on `[t-W_val, t)`, test on `[t, t+W_test)`. Strict monotonicity, asserted by `walk_forward_monotonicity` diagnostic.
- SCE re-fit per fold (its rolling cross-fit is *inside* the training window of each fold; the walk-forward is *outside* it — two nested temporal guards).
- No shuffling anywhere. SEED fixed per fold for reproducibility.

### 8.2 Metrics per horizon
- **RMSE** and **MAE** of predicted vs realised `ret_hN`.
- **Directional hit-rate**: sign(pred) == sign(realised), averaged over test rows.
- **Sharpe and Sortino** of a simple long/short rule: for each test period, long the top decile of `pred_hN` and short the bottom decile (equal-weighted within deciles, dollar-neutral), compute daily PnL = mean(long returns) - mean(short returns); annualise by `sqrt(252)`.

### 8.3 Horizon selection
- For each `(universe, regime_period)` select the horizon `hN*` with the highest walk-forward Sharpe. Record in run metadata alongside the SCE diagnostics results. The chosen horizon is a *reported output*, not an input.

### 8.4 Paper trading (live/streaming, NO real capital)
- Module: `equity/paper_trader/loop.py`.
- Subscribes to a live OHLCV stream (yfinance 1-min/5-min aggregated to daily close) and a live news feed (RSS or a news API).
- Each trading day at close: ingest → sentiment → features (lagged) → SCE `transform` (engine fitted on the rolling window ending at the previous close) → forecast → emit a paper position sheet (long/short deciles) and mark-to-market the previous day's sheet.
- Records every decision in an append-only ledger: `{ts, ticker, action, pred_hN, price, paper_pnl}`.
- NO orders, NO broker client, NO real capital. This is the same rule as SCE's "test set touched exactly once" — the paper-trading ledger is the test set.

### 8.5 Run metadata & reproducibility
- Mirrors SCE's `metadata.json` (`git SHA + config hash + seed + run_grade`). Every backtest and paper-trading run is traceable to an exact commit and config. `run_grade` flags: `exploratory | diagnostic | report-grade`.

## 9. Risks & leakage-safety (finance-specific)

| # | Risk | Mitigation / assertion |
|---|---|---|
| 9.1 | **Look-ahead in indicator windows** (e.g. moving average includes today's close) | `lookahead_indicator` diagnostic recomputes each feature from `prices[:t]` and asserts equality. |
| 9.2 | **Future news bleeding into a period** (text↔price misalignment) | `published_at <= period_close(P)` enforced at the join; `published_at_guard` diagnostic. |
| 9.3 | **Forward target leaking as a feature** | `forward_target_isolation` diagnostic; targets `ret_hN` are kept on a separate column set, never passed to SCE or to feature builders. |
| 9.4 | **Survivorship bias** (universe = today's listed tickers) | Delisting-aware universe construction; `survivorship_check` diagnostic asserts delisted tickers present. |
| 9.5 | **Non-stationarity / regime change** | Rolling cross-fit + rolling walk-forward; `time_bucket` SCE features expose regime; horizon selection per regime; report metrics per regime period, not just global. |
| 9.6 | **Overfitting a backtest** (repeated runs until Sharpe looks good) | One config → one run; horizon selection done on validation, never on test; metrics frozen at report-grade; `permuted_target` must show no SCE advantage on shuffled returns. |
| 9.7 | **Sentiment model drift / latency** in live phase | FinBERT for backtest; pluggable classifier interface; lighter VADER fallback for paper-trading; classifier choice recorded in metadata. |
| 9.8 | **High-cardinality group blow-up** in SCE interactions | Curated interaction allow-list (§6.5 extension 1); `min_group_size=20` back-off to global stats. |
| 9.9 | **Residual layer leaking sector-head in-sample predictions** | Sector-head predictions used as residual targets are computed OOF (cross-fitted), exactly mirroring SCE's own OOF protocol. |
| 9.10 | **Time-zone / market-holiday misalignment** between text and prices | `period_close_ts` is the exchange-local close; holidays from a market-holiday calendar (`pandas_market_calendars`); articles on holidays roll to the next trading period. |

## 10. Proposed vertical slices (dependency order, independently shippable)

Each slice is a PR-sized increment that compiles, tests green, and delivers value on its own. Later slices depend on earlier via `blocked by`.

- **S1 — Data acquisition + canonical schema.** Ship `EquityDataLoader` (yfinance + Kaggle historical), delisting-aware universe, parquet stores, `(ticker, period_close_ts)` canonical schema, point-in-time text join, `published_at_guard` diagnostic. **Value:** a leakage-safe, reproducible equity dataset that any downstream model can consume.
- **S2 — FinBERT sentiment + per-period aggregation.** Ship `FinBERTSententer` with cached, idempotent scoring and time-decayed count-weighted per-`(ticker,period)` aggregation. **Value:** sentiment features with provable point-in-time alignment, reusable across any forecasting model.
- **S3 — Lag-aware technical + sentiment feature layer.** Ship `FeatureConfig`, all technical indicators, lag windows, the `lookahead_indicator` recomputation guard. **Value:** a strictly past-only feature block, usable independently of SCE and the forecaster.
- **S4 — SCE equity enrichment (rolling cross-fit).** Ship the `equity/sce/enrich.py` wrapper, the equity hierarchy config, the curated interaction allow-list (engine extension 1 if needed), `walk_forward_monotonicity` and `survivorship_check` diagnostics, and reuse of `permuted_target` / `shuffled_groups` / `crossfit_ab` / feature-dominance on the equity target. **Value:** SCE's leakage-safe hierarchical context, proven on a financial regime — the core research contribution of this epic.
- **S5 — Multi-horizon two-layer forecaster (offline train + predict).** Ship the sector-head + instrument-residual model pair per horizon, OOF residual labels, the `forward_target_isolation` diagnostic, and a single-fold train/test run on the historical corpus. **Value:** a buildable, leakage-safe forecaster producing `pred_hN` for all configured horizons.
- **S6 — Walk-forward backtest + metrics.** Ship the walk-forward runner, per-horizon RMSE/MAE/hit-rate, decile long/short Sharpe/Sortino, horizon selection on validation, run metadata, and the baseline-vs-SCE comparison report. **Value:** the report-grade numbers that prove (or refute) SCE's value on equities, fully reproducible.
- **S7 — Paper-trading loop.** Ship the live stream ingestion, weekly SCE refit, online `transform`, paper position sheets, append-only ledger, mark-to-market, and the FinBERT/VADER classifier swap. **Value:** a no-capital live validation harness, ready to hand off to the future trading-bot epic.
- **S8 — Diagnostics & report-grade gate for equity.** Ship the consolidated equity diagnostics gate (all of §6.3 + §6.4), the report-grade promotion wired into the run metadata, and the paper figures (mirroring SCE's `docs/figures/paper/`). **Value:** the credibility layer — no equity number is citable until it passes the same gate SCE enforces on its existing benchmarks.

Dependency graph: S1 ← S2 ← S3 ← S4 ← S5 ← S6 ← S7 ; S8 depends on S4 and S6 and can run in parallel with S7.

## 11. Open questions

- **Q1.** Universe size for the report-grade run: S&P 500 (large, liquid, well-covered by FinBERT-relevant news) vs a broader Russell 1000 / all-US-equities universe. Larger universe stresses SCE's high-cardinality handling but increases survivorship-bias remediation cost. Recommend S&P 500 for the first report-grade; broaden later.
- **Q2.** Should the SCE target be `ret_1d` (one-day realised return at period close) or a multi-day realised return (`ret_5d` past)? `ret_1d` is noisier but strictly point-in-time; `ret_5d` past overlaps with future periods for horizons ≤ 5. Recommend `ret_1d` to avoid any overlap with `ret_h1`.
- **Q3.** News source for the live paper-trading phase: which free/low-latency feed is acceptable? (RSS vs a paid API.) This is a cost decision for Mateusz.
- **Q4.** Are the two proposed SCE engine extensions (interaction allow-list, `transform_partial`) acceptable as upstream changes to `sce/`, or should they live in `equity/sce/` as wrappers? Recommend ADR for at least the interaction allow-list.
- **Q5.** Horizon set `{1, 5, 10, 21}` — is 21 the longest horizon we want, or should a 63-day (quarterly) head be included for the regime/macro view?
- **Q6.** Walk-forward fold geometry: 5y train / 63d test is conservative; a 3y train would yield more test folds but a shorter regime window. Confirm the trade-off for the report-grade run.
- **Q7.** Does the future trading-bot epic need this pipeline to expose per-horizon *prediction intervals* (not just point forecasts)? If yes, S5 should add quantile heads (LightGBM quantile regression) — small extra cost now, expensive to retrofit later.

---

**Sign-off:** This PRD is ready for the GATE 1 HITL review with Mateusz. Decisions needed on Q1–Q7 (collected, per global rule 1) before `spec-review` and `decomposer` proceed.