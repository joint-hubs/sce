"""
@module: tests.equity.test_fetch_prices
@depends: equity.data.loader, equity.data.fetch, equity.data.schema, data.download
@exports:
@data_flow: synthetic universe -> EquityDataLoader.fetch_prices -> partitioned parquet -> read-back
"""

from __future__ import annotations

import pandas as pd
import pytest

from equity.data.loader import EquityDataLoader
from equity.data.schema import CANONICAL_PRICE_COLUMNS
from tests.equity.conftest import _kaggle_available, _network_or_yfinance_available

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
    df["hlc_average"] = (df["high"] + df["low"] + df["close"]) / 3.0
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
    removed, not merged. Uses a path INSIDE PROJECT_ROOT (with the
    ``.equity_store`` marker) because the containment guard refuses to rmtree
    paths outside the repo."""
    from equity.data.fetch import STORE_MARKER
    from equity.data.registry import PROJECT_ROOT

    _write_synthetic_universe(tmp_path)
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "equity.data.fetch.fetch_yfinance_ohlcv", _fake_ohlcv
    )

    # Use a path inside PROJECT_ROOT (so the rmtree guard permits the rewrite)
    # and write the sentinel marker so an existing dir is recognized as ours.
    out_dir = PROJECT_ROOT / "data" / "equity" / "_test_stale_prices"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / STORE_MARKER).write_text("marker", encoding="utf-8")
    # Seed a stale partition that the fake would never write.
    stale = out_dir / "year=1999" / "month=12"
    stale.mkdir(parents=True)
    (stale / "stale.parquet").write_bytes(b"NOT PARQUET -- stale sentinel")

    try:
        loader = EquityDataLoader("synthetic", "2024-01-01", "2024-02-28")
        loader.fetch_prices(output_dir=out_dir)

        assert not stale.exists(), "stale partition should have been removed"
        # And valid 2024 partitions exist.
        assert (out_dir / "year=2024" / "month=1").exists()
    finally:
        import shutil as _sh

        if out_dir.exists():
            _sh.rmtree(out_dir)


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
# M1: yfinance end-exclusivity off-by-one (+1 day in fetch_yfinance_ohlcv).
# ---------------------------------------------------------------------------


class _RecordingTicker:
    """Fake yfinance.Ticker that records the (start, end) string args passed
    to ``.history`` and returns a tiny deterministic frame spanning those
    dates (inclusive of the ``start``, exclusive of ``end`` -- mirroring real
    yfinance semantics). Used to verify the fetcher offsets ``end`` by +1 day.
    """

    calls: list[tuple[str, str]] = []

    def __init__(self, symbol: str):
        self.symbol = symbol

    def history(self, start=None, end=None, interval="1d", **_kwargs):
        type(self).calls.append((start, end))
        # Mirror real yfinance: [start, end) exclusive on end.
        dates = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
        if len(dates) == 0:
            return pd.DataFrame()
        idx = pd.DatetimeIndex(dates).tz_localize("America/New_York")
        n = len(idx)
        return pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [101.0] * n,
                "Low": [99.0] * n,
                "Close": [100.5] * n,
                "Adj Close": [100.5] * n,
                "Volume": [1_000_000.0] * n,
            },
            index=idx,
        )


def test_fetch_yfinance_ohlcv_offsets_end_by_one_day(monkeypatch):
    """The loader's `end` is inclusive, but yfinance treats `end` as
    exclusive -- the fetcher must offset by +1 day so the last window day is
    returned. Verifies the actual yfinance call receives end+1d and the
    returned frame contains the inclusive last day."""
    import equity.data.fetch as fetch_mod

    _RecordingTicker.calls = []
    monkeypatch.setattr(fetch_mod.yf, "Ticker", _RecordingTicker)
    df = fetch_mod.fetch_yfinance_ohlcv(
        ["AAPL"], "2024-01-01", "2024-01-31", period="1d"
    )
    # The fetcher must have passed end_str = 2024-02-01 (+1 day) to yfinance.
    assert _RecordingTicker.calls, "yfinance.Ticker.history was never called"
    assert _RecordingTicker.calls[0][1] == "2024-02-01", (
        f"expected end_str='2024-02-01' (inclusive end +1d), got "
        f"{_RecordingTicker.calls[0][1]}"
    )
    # And the inclusive last day (2024-01-31) is present in the output.
    last = pd.Timestamp(df["period_close_ts"].max())
    assert last.date() == pd.Timestamp("2024-01-31").date(), (
        f"expected last session date 2024-01-31 (inclusive), got {last.date()}"
    )


# ---------------------------------------------------------------------------
# M1 spot-check: canonicalized period_close_ts hour != 00 (B1).
# ---------------------------------------------------------------------------


def test_fetch_yfinance_ohlcv_canonicalizes_index_hour_away_from_midnight(monkeypatch):
    """yfinance returns 00:00 ET index values; the fetcher must canonicalize
    them to the XNYS session close (16:00 / 13:00 ET) -- i.e. the hour of
    every returned ``period_close_ts`` must NOT be 00."""
    import equity.data.fetch as fetch_mod

    monkeypatch.setattr(fetch_mod.yf, "Ticker", _RecordingTicker)
    df = fetch_mod.fetch_yfinance_ohlcv(
        ["AAPL"], "2024-07-01", "2024-07-08", period="1d"
    )
    assert not df.empty
    hours = df["period_close_ts"].dt.hour.unique()
    assert 0 not in hours, (
        f"canonicalization failed: 00:00 ET index values leaked through; "
        f"hours={sorted(hours.tolist())}"
    )
    # 2024-07-03 is an early-close (13:00 ET); the rest are 16:00 ET.
    assert set(hours.tolist()).issubset({13, 16})


# ---------------------------------------------------------------------------
# M2: destructive rmtree containment guard.
# ---------------------------------------------------------------------------


def test_fetch_prices_refuses_rmtree_outside_project_root(tmp_path, monkeypatch):
    """A misconfigured absolute output_dir outside PROJECT_ROOT must NOT be
    rmtree'd -- the loader raises ValueError before any deletion."""
    _write_synthetic_universe(tmp_path)
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "equity.data.fetch.fetch_yfinance_ohlcv", _fake_ohlcv
    )

    # Create a pre-existing directory OUTSIDE the repo (in tmp_path root is
    # outside PROJECT_ROOT since tmp_path is the OS temp dir).
    outside = tmp_path / "outside_repo" / "prices"
    outside.mkdir(parents=True)
    # Plant a sentinel file that must NOT be deleted.
    canary = outside / "CANARY_DO_NOT_DELETE.txt"
    canary.write_text("must survive", encoding="utf-8")

    loader = EquityDataLoader("synthetic", "2024-01-01", "2024-02-28")
    # The fake _fake_ohlcv produces data; the loader will try to rmtree the
    # existing outside path -> must raise ValueError BEFORE deleting.
    with pytest.raises(ValueError, match="outside PROJECT_ROOT"):
        loader.fetch_prices(output_dir=outside)
    assert canary.exists(), "canary file was deleted despite the guard"


def test_fetch_prices_refuses_rmtree_without_marker(tmp_path, monkeypatch):
    """An existing directory inside PROJECT_ROOT but WITHOUT the
    ``.equity_store`` marker must NOT be rmtree'd -- refuse with ValueError."""
    _write_synthetic_universe(tmp_path)
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "equity.data.fetch.fetch_yfinance_ohlcv", _fake_ohlcv
    )

    # Build a directory inside PROJECT_ROOT without the sentinel marker.
    # Use the actual PROJECT_ROOT via the loader's resolution: write to a
    # path the loader resolves under PROJECT_ROOT but pre-create WITHOUT the
    # marker.
    from equity.data.registry import PROJECT_ROOT

    target = PROJECT_ROOT / "data" / "equity" / "_test_no_marker"
    target.mkdir(parents=True, exist_ok=True)
    canary = target / "CANARY.txt"
    canary.write_text("survive", encoding="utf-8")
    try:
        loader = EquityDataLoader("synthetic", "2024-01-01", "2024-02-28")
        with pytest.raises(ValueError, match="missing sentinel marker"):
            loader.fetch_prices(output_dir=target)
        assert canary.exists(), "canary was deleted despite missing marker"
    finally:
        # Cleanup the canary we created under PROJECT_ROOT.
        import shutil as _sh

        if target.exists():
            _sh.rmtree(target)


# ---------------------------------------------------------------------------
# M3: _meta.json provenance + frozen flag.
# ---------------------------------------------------------------------------


def test_fetch_prices_writes_meta_json(tmp_path, monkeypatch):
    _write_synthetic_universe(tmp_path)
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "equity.data.fetch.fetch_yfinance_ohlcv", _fake_ohlcv
    )

    out_dir = tmp_path / "prices"
    loader = EquityDataLoader("synthetic", "2024-01-01", "2024-02-28")
    out = loader.fetch_prices(output_dir=out_dir)

    meta_path = out / "_meta.json"
    assert meta_path.exists(), f"_meta.json not written at {meta_path}"
    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["kind"] == "prices"
    assert meta["period"] == "1d"
    assert meta["row_count"] > 0
    assert "content_sha256" in meta and len(meta["content_sha256"]) == 64
    assert "fetched_at_utc" in meta
    # fetched_at_utc must be a tz-aware UTC ISO string (NOT utcnow() naive).
    assert pd.Timestamp(meta["fetched_at_utc"]).tz is not None
    assert meta.get("yfinance_version")


def test_fetch_prices_frozen_refuses_refetch(tmp_path, monkeypatch):
    _write_synthetic_universe(tmp_path)
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", tmp_path)
    monkeypatch.setattr(
        "equity.data.fetch.fetch_yfinance_ohlcv", _fake_ohlcv
    )

    out_dir = tmp_path / "prices"
    loader = EquityDataLoader("synthetic", "2024-01-01", "2024-02-28")
    out = loader.fetch_prices(output_dir=out_dir)
    meta_path = out / "_meta.json"
    assert meta_path.exists()
    first_mtime = meta_path.stat().st_mtime_ns

    # Second call with frozen=True must NOT rewrite (meta untouched).
    out2 = loader.fetch_prices(output_dir=out_dir, frozen=True)
    assert out2 == out
    assert meta_path.stat().st_mtime_ns == first_mtime, (
        "frozen flag should have prevented re-fetch / meta rewrite"
    )


# ---------------------------------------------------------------------------
# m7: fetch_yfinance_ohlcv error-path tests.
# ---------------------------------------------------------------------------


def test_fetch_yfinance_ohlcv_empty_tickers_returns_empty():
    from equity.data.fetch import fetch_yfinance_ohlcv

    df = fetch_yfinance_ohlcv([], "2024-01-01", "2024-01-31", period="1d")
    assert df.empty
    assert list(df.columns) == CANONICAL_PRICE_COLUMNS


def test_fetch_yfinance_ohlcv_all_tickers_fail_returns_empty(monkeypatch):
    """When every yfinance call raises, the batch returns an empty canonical
    frame (does not crash on a single ticker failure)."""
    import equity.data.fetch as fetch_mod

    class _BoomTicker:
        def __init__(self, symbol):
            pass

        def history(self, **_kwargs):
            raise RuntimeError("simulated yfinance failure")

    monkeypatch.setattr(fetch_mod.yf, "Ticker", _BoomTicker)
    df = fetch_mod.fetch_yfinance_ohlcv(
        ["AAPL", "MSFT"], "2024-01-01", "2024-01-31", period="1d"
    )
    assert df.empty
    assert list(df.columns) == CANONICAL_PRICE_COLUMNS


def test_fetch_yfinance_ohlcv_none_history_warns_and_skips(monkeypatch, caplog):
    """A ``None`` history (possible masked fetch error) is logged at WARNING
    and the ticker is skipped (not silently treated as 'no data')."""
    import logging

    import equity.data.fetch as fetch_mod

    class _NoneTicker:
        def __init__(self, symbol):
            pass

        def history(self, **_kwargs):
            return None

    monkeypatch.setattr(fetch_mod.yf, "Ticker", _NoneTicker)
    with caplog.at_level(logging.WARNING, logger="equity.data.fetch"):
        df = fetch_mod.fetch_yfinance_ohlcv(
            ["AAPL"], "2024-01-01", "2024-01-31", period="1d"
        )
    assert df.empty
    assert any("returned None" in rec.message for rec in caplog.records), (
        "expected a WARNING about None history (m5: don't mask silently)"
    )


# ---------------------------------------------------------------------------
# Gated integration tests (skip on default run).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _network_or_yfinance_available(),
    reason="Set SCE_EQUITY_LIVE_TEST=1 to run the yfinance live integration test",
)
def test_fetch_prices_yfinance_30day_slice(tmp_path):
    """Live: pull a 30-day yfinance slice for one ticker and assert PK unique
    and that the index hour was canonicalized away from 00:00 ET (B1)."""
    from equity.data.fetch import fetch_yfinance_ohlcv

    start = "2024-01-02"
    end = "2024-02-01"  # ~30 calendar days
    df = fetch_yfinance_ohlcv(["AAPL"], start, end, period="1d")
    assert not df.empty, "yfinance returned no rows for AAPL (network/source issue?)"
    assert list(df.columns) == CANONICAL_PRICE_COLUMNS
    assert df["period_close_ts"].dt.tz is not None
    # B1: the yfinance 00:00 ET index must have been canonicalized to the XNYS
    # session close -- the hour of every period_close_ts must NOT be 00.
    hours = df["period_close_ts"].dt.hour.unique()
    assert 0 not in hours, (
        f"canonicalization failed: 00:00 ET values leaked through; hours="
        f"{sorted(hours.tolist())}"
    )
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
    df["hlc_average"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df = df[CANONICAL_PRICE_COLUMNS]

    from equity.data.schema import validate_prices

    validated = validate_prices(df)
    dups = validated.duplicated(subset=["ticker", "period_close_ts"]).sum()
    assert dups == 0, f"PK has {dups} duplicates after normalization"
