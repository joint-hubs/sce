# equity.data — OHLCV ingestion + parquet stores

This subpackage implements slice S1 of the equity pipeline:

- **S1.1** delisting-aware universe registry (`registry.py`, `loader.py`)
- **S1.2** OHLCV ingestion + pandera-validated partitioned parquet (`schema.py`,
  `fetch.py`, `EquityDataLoader.fetch_prices`) — this file documents S1.2.
- **S1.3** (planned) point-in-time text join with `published_at_guard`.

## Canonical schema (`prices.parquet`)

Nine data columns, in order (see `equity.data.schema.CANONICAL_PRICE_COLUMNS`):

| column            | dtype                                  | nullable | notes                                                |
|-------------------|----------------------------------------|----------|------------------------------------------------------|
| `ticker`          | `str`                                  | no       | e.g. `AAPL`; may contain dots (`BRK.B`)              |
| `period_close_ts` | `datetime64[ns, America/New_York]`      | no       | **tz-aware** exchange-local session close (16:00 ET) |
| `open`            | `float`                                | yes      | NaN on partial bars (delisted within window)         |
| `high`            | `float`                                | yes      |                                                      |
| `low`             | `float`                                | yes      |                                                      |
| `close`           | `float`                                | yes      |                                                      |
| `adj_close`       | `float`                                | yes      | yfinance `Adj Close` (requires `auto_adjust=False`)  |
| `volume`          | `float`                                | yes      |                                                      |
| `vwap`            | `float`                                | yes      | proxy `(high+low+close)/3` (yfinance has no daily VWAP) |

Primary key: `(ticker, period_close_ts)` — enforced by
`equity.data.schema.assert_primary_key_unique`.

On read-back from the partitioned parquet store, two additional columns
`year` and `month` are reconstructed from the Hive partition path (see
"Partition layout" below). They are partition metadata, not data columns;
downstream code should not rely on them as OHLCV fields.

### Timezone semantics (PRD requirement)

`period_close_ts` is the **exchange-local session close** (16:00 ET for US
daily bars), stored as a tz-aware `America/New_York` timestamp. yfinance
returns an `America/New_York`-aware `DatetimeIndex`; we preserve it verbatim
and never convert to UTC. Tz-naive values are rejected at validation time
(the pandera column dtype is `pd.DatetimeTZDtype(tz="America/New_York")` with
`coerce=False`, so a tz-naive `datetime64[ns]` Series fails the dtype check).
This is the guard that prevents accidental UTC writes that would break the
point-in-time text join in S1.3.

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
partitions from prior runs with different windows do not leak in.

Round-trip:
```python
from equity.data.loader import EquityDataLoader
loader = EquityDataLoader("sp500", "2024-01-01", "2024-12-31")
out = loader.fetch_prices()               # writes data/equity/prices/
df = pd.read_parquet(out)                 # 11 cols: 9 canonical + year + month
```

## VWAP proxy

yfinance does not return VWAP for daily bars. We compute the standard proxy
`vwap = (high + low + close) / 3` in `equity.data.fetch._add_vwap`. The
result is `NaN` where any of `high`/`low`/`close` is `NaN` (partial bars for
tickers delisted within the window). If a future source provides a true
VWAP, replace `_add_vwap`.

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

- `period_close_ts` is tz-aware `America/New_York` session close; S1.3's
  `published_at` should be compared in the same tz (or normalized) to avoid
  off-by-one joins across DST transitions.
- Read-back frame has 11 columns (9 canonical + `year` + `month` partition
  keys); S1.3 should select the 9 canonical columns explicitly.
- The alive-ticker filter (`loader.universe()`) is the same set S1.3 will
  join against; delisted-during-window tickers are included with partial
  bars (NaN OHLCV), which S1.3 must tolerate.
- Historical source for S1.3's backfill is the Cam Nugent dataset
  (`camnugent/sandp500`); the gated test in this slice exercises the
  normalization path S1.3 will reuse.
