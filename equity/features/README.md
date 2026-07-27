# equity.features — S3 lag-aware technical + sentiment feature layer

Strictly past-only feature block, usable independently of SCE and the forecaster.

## Schema of the feature matrix

`build_features(prices, sentiment_per_period=None, ...)` returns a **flat long-form** DataFrame: one row per `(ticker, period_close_ts)`. Columns:

| Group | Columns | Source |
|---|---|---|
| Price (passthrough) | `ticker, period_close_ts, open, high, low, close, adj_close, volume, hlc_average` | S1 canonical schema (`equity/data/schema.py`) |
| Technical indicators | `ret_{1,5,10,21}d_log`, `sma_{5,10,21,63}`, `ema_{5,10,21,63}`, `rsi_14`, `macd, macd_signal, macd_hist`, `volatility_{21,63}`, `volume_zscore_21`, `atr_14`, `bb_mid, bb_upper, bb_lower` | `equity/features/technical.py` |
| Sentiment (LEFT-JOIN) | `sentiment_score, sentiment_pos, sentiment_neg, sentiment_neu, n_articles` | FOC-49 `aggregate_per_period` output |
| Lag layer | `{base_col}_lag{N}`, `{base_col}_rollmean{N}`, `{base_col}_rollstd{N}` for `N in (1, 3, 5, 10, 21)` over each technical + sentiment base col | `equity/features/lag.py` |

`period_close_ts` is tz-aware. Prices arrive as `America/New_York`; `build_features` canonicalizes to UTC before the sentiment LEFT-JOIN (mirrors `equity/data/loader.py:943-949`). The output `period_close_ts` is in UTC.

## PIT semantics — past-only (D6 / FOOTGUN #1)

EVERY feature at row `t` is a function of `prices[:t]` ONLY (rows strictly before `t` within the same ticker). Concretely:

- **Indicators**: each naive (current-row-inclusive) indicator is `.shift(1)` within the ticker group, so the value stored at row `t` reflects data through `close[t-1]`.
- **Lag layer**:
  - `_lag{N}` = `base.shift(N)` — inherently past-only (value N rows back).
  - `_rollmean{N}` / `_rollstd{N}` use `rolling(N, closed='left')` EXPLICITLY — excludes the current row. The pandas default `closed='right'` (includes current row) is the exact leakage the `lookahead_indicator` guard exists to catch.

### Window units = trading rows, NOT calendar days

The `N` in `ret_{N}d_log`, `sma_{N}`, `ema_{N}`, and the lag windows (`_lag{N}` / `_rollmean{N}` / `_rollstd{N}`) counts **N trading rows (sessions)**, not calendar days. Weekend/holiday gaps are skipped by row-position: if a ticker has sessions Mon, Tue, Thu, Fri, then `sma_2` at Friday = mean of Thursday and Friday closes (the two most-recent sessions), ignoring the Wed gap. This matches the per-ticker `groupby("ticker")` rolling semantics in pandas.

### Sentiment zero-fill disambiguation (`has_sentiment`)

When `sentiment_per_period` is provided, missing `(ticker, period)` rows in the aggregate are LEFT-JOINed as NaN and then filled to `0` for `sentiment_score` / `sentiment_pos` / `sentiment_neg` / `sentiment_neu` / `n_articles` (D4: missing period = 0 articles, hence neutral score). After the fill, `sentiment_score=0` is ambiguous (genuinely-neutral scored articles vs no articles at all).

To disambiguate, `build_features` adds a `has_sentiment` **bool** column:

- `True` where the LEFT-JOIN matched a non-null sentiment row AND `n_articles > 0` (after the fill).
- `False` where the join missed (no articles for that period) OR `n_articles == 0`.
- For the empty-sentiment-frame branch: `False` for all rows.
- When `sentiment_per_period` is `None` (no sentiment block runs): `has_sentiment` is NOT added.

Downstream consumers SHOULD gate on `has_sentiment` before interpreting `sentiment_score=0` as neutral.

The `equity/diagnostics/lookahead_indicator.py` guard recomputes each indicator from `prices[:t]` only and asserts equality with the stored feature within `abs=1e-9`.

## Per-ticker invariant (D7)

Every rolling/shift op is applied via `groupby("ticker", sort=False)` (after a defensive ascending sort on `period_close_ts`). A rolling window NEVER bleeds across the ticker boundary.

## NaN in early windows (D5)

Pandas `rolling`/`ewm` emit native NaN during warmup. We do NOT mask. The lookahead guard treats NaN as "undefined at this row", NOT a violation. A row where exactly one of (stored, re-derived) is NaN and the other is finite IS a violation (partial-window bug).

## Open questions

- **VWAP vs `hlc_average`**: the canonical S1 price schema ships `hlc_average` (mean of high/low/close), NOT `vwap`. Per lead decision D2, S3 uses `hlc_average` as the VWAP proxy and does NOT change the canonical schema in this ticket. A future ticket may add a true `vwap` column (requires volume-weighted aggregation in S1) and the technical block can be extended to consume it.
- **Sentiment lag base columns**: `build_features` lags all 5 sentiment cols (`sentiment_score, sentiment_pos, sentiment_neg, sentiment_neu, n_articles`). If downstream proves `n_articles` lags are uninformative, restrict via `lag_base_cols=`.
