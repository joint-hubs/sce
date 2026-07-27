# equity.data — OHLCV ingestion + parquet stores

This subpackage implements slice S1 of the equity pipeline:

- **S1.1** delisting-aware universe registry (`registry.py`, `loader.py`)
- **S1.2** OHLCV ingestion + pandera-validated partitioned parquet (`schema.py`,
  `fetch.py`, `EquityDataLoader.fetch_prices`) — this file documents S1.2.
- **S1.3** point-in-time text join with `published_at_guard` (implemented).

## Canonical schema (`prices.parquet`)

Nine data columns, in order (see `equity.data.schema.CANONICAL_PRICE_COLUMNS`):

| column            | dtype                                  | nullable | notes                                                |
|-------------------|----------------------------------------|----------|------------------------------------------------------|
| `ticker`          | `str`                                  | no       | e.g. `AAPL`; may contain dots (`BRK.B`)              |
| `period_close_ts` | `datetime64[ns, America/New_York]`      | no       | **tz-aware** XNYS session close (16:00 / 13:00 ET)  |
| `open`            | `float`                                | yes      | NaN on partial bars (delisted within window)         |
| `high`            | `float`                                | yes      |                                                      |
| `low`             | `float`                                | yes      |                                                      |
| `close`           | `float`                                | yes      |                                                      |
| `adj_close`       | `float`                                | yes      | yfinance `Adj Close` (requires `auto_adjust=False`)  |
| `volume`          | `float`                                | yes      |                                                      |
| `hlc_average`     | `float`                                | yes      | `(high+low+close)/3` (NOT VWAP; yfinance has no daily VWAP) |

Primary key: `(ticker, period_close_ts)` — enforced by
`equity.data.schema.assert_primary_key_unique`.

On read-back from the partitioned parquet store, two additional columns
`year` and `month` are reconstructed from the Hive partition path (see
"Partition layout" below). They are partition metadata, not data columns;
downstream code should not rely on them as OHLCV fields.

### Timezone semantics (PRD requirement)

`period_close_ts` is the **XNYS session close** (16:00 ET for normal daily
bars, 13:00 ET on early-close sessions), stored as a tz-aware
`America/New_York` timestamp. yfinance returns a `DatetimeIndex` of
**exchange-local midnight (00:00 ET)**, NOT the session close; the fetcher
(`equity.data.fetch._canonicalize_session_close`) replaces each 00:00 ET
index value with the XNYS `market_close` for that date so that
`prices.parquet.period_close_ts` is a valid foreign key consumable by the S1.3
point-in-time join (which derives `period_close_ts` from the XNYS schedule).
Tz-naive values are rejected at validation time (the pandera column dtype is
`pd.DatetimeTZDtype(tz="America/New_York")` with `coerce=False`, so a
tz-naive `datetime64[ns]` Series fails the dtype check). This is the guard
that prevents accidental UTC writes that would break the join in S1.3.

## Partition layout

`EquityDataLoader.fetch_prices()` writes a Hive-style partitioned parquet
dataset:

```
<output_dir>/
  year=2024/
    month=1/
      <uuid>.parquet
    month=2/
      <uuid>.parquet
  year=2025/
    month=1/
      <uuid>.parquet
```

`year` and `month` are derived from `period_close_ts.dt.year` /
`.dt.month` (America/New_York) before writing and used as `partition_cols`
in `df.to_parquet(...)`. They are NOT stored as data columns in the parquet
files; `pd.read_parquet(output_dir)` reconstructs them from the Hive path.

`fetch_prices()` performs a **full rewrite** (not incremental) of
`output_dir`: if the directory exists it is removed before writing, so stale
partitions from prior runs with different windows do not leak in. The
destructive `rmtree` is guarded (see "Containment guard" below): it refuses to
delete paths outside `PROJECT_ROOT` or directories missing the `.equity_store`
sentinel marker. A `_meta.json` file is written next to the store with
provenance (fetch UTC timestamp, yfinance version, row count, sha256 content
hash) so two runs over the same window can be compared for reproducibility
drift (see "Reproducibility (S1 limitation)" below). A `frozen=True` flag
refuses to re-fetch if the store already exists with a `_meta.json` for the
window.

Round-trip:
```python
from equity.data.loader import EquityDataLoader
loader = EquityDataLoader("sp500", "2024-01-01", "2024-12-31")
out = loader.fetch_prices()               # writes data/equity/prices/
df = pd.read_parquet(out)                 # 11 cols: 9 canonical + year + month
```

## HLC average (NOT VWAP)

yfinance does not return VWAP for daily bars. We compute the standard
`(high + low + close) / 3` average in `equity.data.fetch._add_hlc_average` and
store it under the column name **`hlc_average`** (NOT `vwap`). The formula is
a simple average, not a volume-weighted value; the honest column name prevents
S2 feature engineering from misinterpreting it as true VWAP. The result is
`NaN` where any of `high`/`low`/`close` is `NaN` (partial bars for tickers
delisted within the window). If a future source provides a true VWAP, add it
as a separate `vwap` column.

## Containment guard (destructive rmtree)

Both `fetch_prices()` and `fetch_articles()` perform a full rewrite: if the
output directory exists it is removed before writing. To prevent a
misconfigured TOML / CLI `output_dir` from deleting an unrelated tree
(`C:\Users\...`, `~/`, repo root), the rmtree is guarded by
`equity.data.loader._safe_rmtree`:

1. The resolved path must lie **inside `PROJECT_ROOT`** (absolute paths are
   honored verbatim -- a path outside the repo is refused with
   `ValueError`).
2. An existing directory must contain the `.equity_store` sentinel marker
   file (written by the loader on first creation). A non-empty tree without
   the marker is refused -- we treat it as "not ours" and do not delete it.

## Reproducibility (S1 limitation)

yfinance is an unofficial scraper whose historical values (`Adj Close`,
splits, dividends) can be **revised retroactively** between runs. Two calls
to `fetch_prices` over the same `[start, end]` window can therefore produce
different `adj_close` / `volume` rows -- silent data drift propagating into
features and backtests. `auto_adjust=False` preserves raw prices but does not
freeze the data. The S1 mitigations are:

- a `_meta.json` is written next to each parquet store with the fetch UTC
  timestamp, yfinance version, row count and a sha256 content hash, so two
  runs can be compared for drift;
- a `frozen=True` flag on `fetch_prices` / `fetch_articles` refuses to
  re-fetch if the store already exists with a `_meta.json` for the window.

These are **best-effort** mitigations, not a guarantee of reproducibility.
True frozen snapshots (e.g. content-addressed parquet archives, or a swap to
a pinned historical vendor like the Cam Nugent Kaggle snapshot) are S2+ work.

## How to run `fetch_prices`

```python
from equity.data.loader import EquityDataLoader

loader = EquityDataLoader("sp500", "2024-01-01", "2024-12-31")
out = loader.fetch_prices()                       # -> data/equity/prices/
# or override the output dir:
out = loader.fetch_prices(output_dir="data/equity/prices_2024")
```

Configuration lives in `configs/equity/sp500.toml` under `[prices]`:
`output_dir`, `partition_cols`, `source`. The `output_dir` arg to
`fetch_prices()` takes precedence over the TOML value.

The alive-ticker set is derived from `loader.universe()` (delisting-aware):
only tickers whose listing window overlaps `[start, end]` are fetched. A
ticker delisted during the window is included (survivorship-aware) — yfinance
may return partial data near the delisting event, which surfaces as `NaN`
OHLCV fields and is preserved (not dropped).

## Gated integration tests

Two integration tests live under `tests/equity/` and are `@pytest.mark.skipif`-
gated so the default `pytest` run hits no network and needs no credentials:

1. **yfinance 30-day slice** (`tests/equity/test_fetch_prices.py`): pulls a
   30-day daily slice for a small ticker set via `fetch_yfinance_ohlcv` and
   asserts primary-key uniqueness on `(ticker, period_close_ts)`. Gated on
   `SCE_EQUITY_LIVE_TEST=1` (env var) AND yfinance importability. Run with:
   ```bash
   SCE_EQUITY_LIVE_TEST=1 python -m pytest tests/equity/test_fetch_prices.py -k yfinance -q
   ```

2. **Kaggle Cam Nugent historical slice** (`tests/equity/test_fetch_prices.py`):
   downloads `camnugent/sandp500/all_stocks_5yr.csv` via `data.download`
   (Kaggle CLI), normalizes to the canonical schema (adds `adj_close = close`
   and the VWAP proxy, localizes `date` to tz-aware 16:00 ET), validates, and
   asserts PK uniqueness. Gated on `SCE_EQUITY_LIVE_TEST=1` (env var, shared
   with the yfinance gate so the default run never touches the network) AND
   Kaggle credentials (`KAGGLE_USERNAME` / `KAGGLE_KEY` env vars or
   `configs/kaggle.json`) AND `data.download` importability. Run with:
   ```bash
   SCE_EQUITY_LIVE_TEST=1 python -m pytest tests/equity/test_fetch_prices.py -k kaggle -q
   ```

Both gated tests SKIP on a default `pytest` run (no network, no creds needed).

## Notes for S1.3 (point-in-time text join)

S1.3 status: **implemented**. See the new "Point-in-time text join (S1.3)"
section below. The slice ships:

- `articles_schema` / `validate_articles` / `assert_articles_primary_key_unique`
  in `equity.data.schema` (mirrors the prices schema pattern).
- `fetch_articles_from_seed` in `equity.data.fetch` (NO network; live
  RSS/Kaggle ingestion is S7).
- `EquityDataLoader.fetch_articles` + `EquityDataLoader.join_articles_to_prices`
  in `equity.data.loader`.
- `equity.diagnostics.published_at_guard` leakage guard CLI.
- `[articles]` section in `configs/equity/sp500.toml` and a committed seed
  CSV at `configs/equity/articles_seed.csv`.

## Point-in-time text join (S1.3)

The articles layer binds raw news text to price periods so downstream
(slices S2+) cannot accidentally train on information that was not yet
available. The canonical articles frame is 4 data columns:

| column         | dtype                          | nullable | notes                                                       |
|----------------|--------------------------------|----------|-------------------------------------------------------------|
| `ticker`       | `str`                          | no       | e.g. `AAPL`; rows with NaN ticker are dropped before join   |
| `published_at` | `datetime64[ns, UTC]`          | no       | **tz-aware UTC** (canonical storage tz)                     |
| `text`         | `str`                          | yes      | nullable -- some sources ship headline-only payloads        |
| `source`       | `str`                          | no       | e.g. `reuters`, `bloomberg`                                 |

Primary key: `(ticker, published_at, source)` -- dedup the same article from
the same source. Enforced by
`equity.data.schema.assert_articles_primary_key_unique`.

### Canonical storage timezone: UTC

`published_at` is stored as `DatetimeTZDtype(tz="UTC")` everywhere. RSS and
Kaggle feeds publish in UTC; `pd.Timestamp(...).tz_convert("UTC")` is
idempotent for already-UTC timestamps, and `pd.to_datetime(..., utc=True)`
handles mixed-aware / tz-naive ISO strings in the seed loader. We never
store wall-clock ET for `published_at`: the price-side
`period_close_ts` (America/New_York) is converted to UTC for the comparison
(see "Join rule" below), so the inequality is DST-safe -- no wall-clock
arithmetic across DST transitions.

### Join rule

An article is bound to trading period `P` iff:

    period_close(P-1) < published_at <= period_close(P)

where `period_close` is the NYSE session close from
`pandas_market_calendars`'s `"XNYS"` calendar (16:00 ET for normal sessions,
13:00 ET for early-close sessions). The comparison is performed entirely in
UTC: `period_close_ts` (tz-aware `America/New_York`, canonicalized in the
fetcher) is converted to UTC and compared with `published_at` (already UTC).

The bfill index is built **from the prices store's actual
`period_close_ts` values** (sorted, de-duplicated, in UTC) -- NOT from the
XNYS schedule for the loader window. This guarantees every bound
`period_close_ts` is a real key in `prices.parquet` (no phantom foreign
keys). The `published_at_guard` diagnostic includes an FK integrity check
(every joined `period_close_ts` must exist in the prices store) which is
asserted on every CLI run.

### Under-inclusion at window edges

Because the binding index is restricted to sessions that actually exist in
`prices.parquet`, an article published after the last stored session close
(e.g. Friday 17:00 ET when the store covers through Friday's close) is
**dropped** from the joined output. This is **under-inclusion, NOT leakage**
-- no future information is leaked into a feature row; the article simply
finds no period to bind to within the stored coverage. The loader logs the
drop count. If you need edge articles bound, re-fetch prices for a wider
window; do not extend the join window with a schedule-derived +1-day tail
(that produces phantom FKs, which the guard rejects).

Both `period_close_ts` and `published_at` are written to
`articles_joined.parquet` canonicalized to **UTC** (downstream S2+ slices can
compare them without tz juggling).

### Holiday roll-forward

Holiday / weekend roll-forward is implicit in the XNYS schedule: the
calendar simply omits non-trading days. An article published on
2024-07-04 (US Independence Day holiday) binds to the NEXT trading session
(2024-07-05); an article published on 2024-07-06 (Saturday) binds to
2024-07-08 (Monday). No special-case code is needed -- the XNYS schedule
already encodes the NYSE holiday calendar.

### `published_at_guard` leakage diagnostic

`equity/diagnostics/published_at_guard.py` is a **join-invariant sanity
check, not a feed-side leakage detector**. It asserts three invariants on
the joined frame:

1. **Right-hand side** -- `published_at <= period_close_ts` for every row.
   This holds by construction for the bfill-derived join; the check catches
   regressions (e.g. a switch to `method='ffill'`) and downstream tampering.
   It does NOT detect feed-side backdating (a feed that backdates
   `published_at` cannot be caught here -- the value is already a canonical
   UTC timestamp by the time the join runs).
2. **Left-hand side** -- `period_close(P-1) < published_at` (the previous
   stored session close is strictly before `published_at`, OR the row is the
   first stored session). Catches an `ffill` regression where an article is
   wrongly bound to the earlier session.
3. **Foreign-key integrity** (when `--prices` is provided): every
   `period_close_ts` in the joined frame must exist in the prices store.

Run as a module:

```bash
python -m equity.diagnostics.published_at_guard --joined <articles_joined.parquet>
# or, when a pre-joined file does not exist yet:
python -m equity.diagnostics.published_at_guard --prices <prices_dir> --articles <articles_dir>
```

The CLI writes a JSON result (pass / n_violations / violations / n_checked)
under `results/diagnostics/equity/` and exits with code **0 on PASS, 1 on
any violation**. This diverges from the rest of SCE's `scripts/diagnostics/`
family, which rely on uncaught exceptions for failure -- the DoD for S1.3
requires a non-zero exit on a synthetic injected violation.

### S2 hand-off (market-wide sentiment aggregate -- NOT in S1.3)

S1.3 ships only the raw `articles.parquet` store and the point-in-time
join. The market-wide sentiment aggregate (rolling sentiment index across
the universe, used as a feature in S2 model inputs) is **S2 work** and is
NOT implemented here. Downstream slices should consume
`articles_joined.parquet` (ticker, period_close_ts, published_at, text,
source) and compute sentiment features per their own definitions.
