"""
@module: tests.equity.test_fetch_prices
@depends: equity.data.loader, equity.data.fetch, equity.data.schema, data.download
@exports:
@data_flow: synthetic universe -> EquityDataLoader.fetch_prices -> partitioned parquet -> read-back
"""

from __future__ import annotations

import pandas as pd
import pytest

from conftest import _kaggle_available, _network_or_yfinance_available
from equity.data.loader import EquityDataLoader
from equity.data.schema import CANONICAL_PRICE_COLUMNS


# ---------------------------------------------------------------------------
# Helpers: synthetic universe + fake yfinance.
# ---------------------------------------------------------------------------

_SYNTH_TICKERS = ["AAPL", "MSFT", "GOOG"]


def _write_synthetic_universe(tmp_path) -> None:
    """Write a 3-ticker synthetic universe CSV + TOML into tmp_path.

    The TOML uses an ABSOLUTE ``universe_file`` (single-quoted TOML literal so
    Windows backslashes are preserved verbatim) so the loader resolves the
    CSV from ``tmp_path`` rather than ``PROJECT_ROOT``.
    """
    csv = tmp_path / "synthetic_universe.csv"
    csv.write_text(
        "ticker,listed_at,delisted_at,name\n"
        "AAPL,,,Apple\n"
        "MSFT,,,Microsoft\n"
        "GOOG,,,Alphabet\n",
        encoding="utf-8",
    )
    csv_posix = csv.as_posix()
    toml = tmp_path / "synthetic.toml"
    toml.write_text(
        '[universe]\n'
        'name = "synthetic"\n'
        'description = "3-ticker synthetic universe for unit tests"\n'
        'source = "test"\n'
        f"universe_file = '{csv_posix}'\n"  # absolute TOML literal string
        '\n'
        '[prices]\n'
        'output_dir = "data/equity/prices"\n'
        'partition_cols = ["year", "month"]\n'
        'source = "yfinance"\n',
        encoding="utf-8",
    )


def _fake_ohlcv(tickers, start, end, period="1d"):
    """Deterministic synthetic OHLCV frame (no network). Spans Jan + Feb 2024
    so the partitioned parquet write produces multiple year/month partitions.
    """
    dates = pd.bdate_range("2024-01-02", "2024-02-15")
    rows = []
    for i, tk in enumerate(tickers):
        base = 100.0 + i * 10.0
        for d in dates:
            close = base
            rows.append(
                {
                    "ticker": tk,
                    "period_close_ts": d + pd.Timedelta(hours=16),
                    "open": close - 1.0,
                    "high": close + 1.0,
                    "low": close - 2.0,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000_000.0,
                }
            )
    df = pd.DataFrame(rows)
    # Localize naive 16:00 -> tz-aware America/New_York session close.
    df["period_close_ts"] = df["period_close_ts"].dt.tz_localize("America/New_York")
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    return df[CANONICAL_PRICE_COLUMNS]


# ---------------------------------------------------------------------------
# Non-network tests: fetch_prices end-to-end with synthetic data.
# ---------------------------------------------------------------------------


def test_fetch_prices_writes_partitioned_parquet(tmp_path, monkeypatch):
    _write_synthetic_universe(tmp_path)
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "equity.data.fetch.fetch_yfinance_ohlcv", _fake_ohlcv
    )

    out_dir = tmp_path / "prices"
    loader = EquityDataLoader("synthetic", "2024-01-01", "2024-02-28")
    out = loader.fetch_prices(output_dir=out_dir)

    assert out == out_dir
    assert out.exists()
    # Hive-style partition dirs.
    year_dirs = list(out.glob("year=*"))
    assert year_dirs, f"expected year=* partition dirs, got {list(out.iterdir())}"
    month_dirs = list(out.glob("year=*/month=*"))
    assert len(month_dirs) >= 2, (
        f"expected >=2 month partitions (Jan+Feb), got {len(month_dirs)}"
    )


def test_fetch_prices_readback_has_canonical_cols_and_tz(tmp_path, monkeypatch):
    _write_synthetic_universe(tmp_path)
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "equity.data.fetch.fetch_yfinance_ohlcv", _fake_ohlcv
    )

    out_dir = tmp_path / "prices"
    loader = EquityDataLoader("synthetic", "2024-01-01", "2024-02-28")
    out = loader.fetch_prices(output_dir=out_dir)

    df = pd.read_parquet(out)
    # The 9 canonical data columns must be present (year/month are partition
    # metadata reconstructed from the Hive path).
    assert set(CANONICAL_PRICE_COLUMNS).issubset(set(df.columns)), (
        f"missing canonical cols; got {list(df.columns)}"
    )
    # period_close_ts must be tz-aware America/New_York.
    ts = df["period_close_ts"]
    assert ts.dt.tz is not None, "period_close_ts must be tz-aware"
    assert str(ts.dt.tz) == "America/New_York"
    # Hive partition keys are reconstructed on read-back (documented in README).
    assert "year" in df.columns and "month" in df.columns


def test_fetch_prices_primary_key_is_unique(tmp_path, monkeypatch):
    _write_synthetic_universe(tmp_path)
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "equity.data.fetch.fetch_yfinance_ohlcv", _fake_ohlcv
    )

    out_dir = tmp_path / "prices"
    loader = EquityDataLoader("synthetic", "2024-01-01", "2024-02-28")
    out = loader.fetch_prices(output_dir=out_dir)

    df = pd.read_parquet(out)
    dups = df.duplicated(subset=["ticker", "period_close_ts"]).sum()
    assert dups == 0, f"PK (ticker, period_close_ts) has {dups} duplicates"
    # All three synthetic tickers present.
    assert set(df["ticker"]) == set(_SYNTH_TICKERS)


def test_fetch_prices_full_rewrite_clears_stale_partitions(tmp_path, monkeypatch):
    """fetch_prices is a full rewrite: stale partitions from a prior run are
    removed, not merged."""
    _write_synthetic_universe(tmp_path)
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "equity.data.fetch.fetch_yfinance_ohlcv", _fake_ohlcv
    )

    out_dir = tmp_path / "prices"
    # Seed a stale partition that the fake would never write.
    stale = out_dir / "year=1999" / "month=12"
    stale.mkdir(parents=True)
    (stale / "stale.parquet").write_bytes(b"NOT PARQUET -- stale sentinel")

    loader = EquityDataLoader("synthetic", "2024-01-01", "2024-02-28")
    loader.fetch_prices(output_dir=out_dir)

    assert not stale.exists(), "stale partition should have been removed"
    # And valid 2024 partitions exist.
    assert (out_dir / "year=2024" / "month=1").exists()


def test_fetch_prices_empty_universe_raises(tmp_path, monkeypatch):
    """A universe with no alive tickers for the window must raise, not write."""
    csv = tmp_path / "synthetic_universe.csv"
    csv.write_text(
        "ticker,listed_at,delisted_at,name\n"
        "OLDCO,2000-01-01,2001-01-01,Delisted early\n",
        encoding="utf-8",
    )
    csv_posix = csv.as_posix()
    toml = tmp_path / "synthetic.toml"
    toml.write_text(
        f"[universe]\nname=\"synthetic\"\nuniverse_file='{csv_posix}'\nsource=\"test\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "equity.data.fetch.fetch_yfinance_ohlcv", _fake_ohlcv
    )

    loader = EquityDataLoader("synthetic", "2024-01-01", "2024-01-31")
    with pytest.raises(ValueError, match="No alive tickers"):
        loader.fetch_prices(output_dir=tmp_path / "prices")


# ---------------------------------------------------------------------------
# Gated integration tests (skip on default run).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _network_or_yfinance_available(),
    reason="Set SCE_EQUITY_LIVE_TEST=1 to run the yfinance live integration test",
)
def test_fetch_prices_yfinance_30day_slice(tmp_path):
    """Live: pull a 30-day yfinance slice for one ticker and assert PK unique."""
    from equity.data.fetch import fetch_yfinance_ohlcv

    start = "2024-01-02"
    end = "2024-02-01"  # ~30 calendar days
    df = fetch_yfinance_ohlcv(["AAPL"], start, end, period="1d")
    assert not df.empty, "yfinance returned no rows for AAPL (network/source issue?)"
    assert list(df.columns) == CANONICAL_PRICE_COLUMNS
    assert df["period_close_ts"].dt.tz is not None
    dups = df.duplicated(subset=["ticker", "period_close_ts"]).sum()
    assert dups == 0, f"PK has {dups} duplicates"


@pytest.mark.skipif(
    not _kaggle_available(),
    reason="Kaggle credentials + data.download required for the historical integration test",
)
def test_fetch_prices_kaggle_camnugent_historical(tmp_path):
    """Live: download the Cam Nugent sandp500 historical CSV, normalize to the
    canonical schema, validate, and assert PK uniqueness."""
    from data.download import download_kaggle_file, parse_source

    spec = parse_source("kaggle://datasets/camnugent/sandp500/all_stocks_5yr.csv")
    dest = tmp_path / "all_stocks_5yr.csv"
    download_kaggle_file(spec, dest)
    assert dest.exists(), "Kaggle download did not produce the expected file"

    raw = pd.read_csv(dest)
    # Cam Nugent columns: date, open, high, low, close, volume, Name (no adj_close).
    required = {"date", "open", "high", "low", "close", "volume", "Name"}
    assert required.issubset(set(raw.columns)), (
        f"unexpected Cam Nugent schema: {list(raw.columns)}"
    )

    # Normalize to canonical schema. session close = date + 16:00 ET.
    naive_close = pd.to_datetime(raw["date"]) + pd.Timedelta(hours=16)
    df = pd.DataFrame(
        {
            "ticker": raw["Name"].astype(str),
            "period_close_ts": naive_close.dt.tz_localize("America/New_York"),
            "open": raw["open"].astype(float),
            "high": raw["high"].astype(float),
            "low": raw["low"].astype(float),
            "close": raw["close"].astype(float),
            "adj_close": raw["close"].astype(float),  # Cam Nugent has no adj_close
            "volume": raw["volume"].astype(float),
        }
    )
    df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df = df[CANONICAL_PRICE_COLUMNS]

    from equity.data.schema import validate_prices

    validated = validate_prices(df)
    dups = validated.duplicated(subset=["ticker", "period_close_ts"]).sum()
    assert dups == 0, f"PK has {dups} duplicates after normalization"
