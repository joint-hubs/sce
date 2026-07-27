"""
@module: tests.equity.test_articles
@depends: equity.data.schema, equity.data.fetch, equity.data.loader,
          equity.diagnostics.published_at_guard, pandas, pandera
@exports:
@data_flow: synthetic frames -> validate_articles / join_articles_to_prices /
            run_published_at_guard / published_at_guard CLI
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity.data.registry import PROJECT_ROOT
from equity.data.schema import (
    ARTICLE_TZ,
    CANONICAL_ARTICLE_COLUMNS,
    articles_schema,
    assert_articles_primary_key_unique,
    validate_articles,
)
from equity.data.fetch import fetch_articles_from_seed
from equity.data.loader import EquityDataLoader
from equity.diagnostics.published_at_guard import run_published_at_guard

SEED_PATH = PROJECT_ROOT / "configs" / "equity" / "articles_seed.csv"


def _valid_articles_frame(n: int = 3) -> pd.DataFrame:
    """Return a small valid articles frame (tz-aware UTC published_at)."""
    pub = pd.date_range("2024-07-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "NVDA"][:n] + ["AAPL"] * max(0, n - 3),
            "published_at": pub,
            "text": [f"article {i}" for i in range(n)],
            "source": ["reuters"] * n,
        }
    )


# ---------------------------------------------------------------------------
# (a)-(d): schema validation
# ---------------------------------------------------------------------------


def test_validate_articles_accepts_valid_utc_aware_frame():
    df = _valid_articles_frame()
    out = validate_articles(df)
    assert list(out.columns) == CANONICAL_ARTICLE_COLUMNS
    assert out["published_at"].dt.tz is not None
    assert str(out["published_at"].dt.tz) == "UTC"


def test_validate_articles_rejects_tz_naive_published_at():
    df = _valid_articles_frame()
    naive = df["published_at"].dt.tz_localize(None)
    df = df.assign(published_at=naive)
    with pytest.raises(Exception, match="UTC"):
        validate_articles(df)


def test_validate_articles_rejects_extra_column_strict():
    df = _valid_articles_frame()
    df["unexpected"] = 1
    with pytest.raises(Exception):
        validate_articles(df)


def test_assert_articles_primary_key_unique_catches_dup():
    df = _valid_articles_frame(n=2)
    # Force a duplicate (ticker, published_at, source) by cloning the first row.
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        assert_articles_primary_key_unique(dup)


def test_articles_schema_is_strict_and_utc_dtype():
    assert articles_schema.strict is True
    col = articles_schema.columns["published_at"]
    assert "UTC" in str(col.dtype), f"unexpected dtype: {col.dtype}"


def test_article_tz_constant_is_utc():
    assert ARTICLE_TZ == "UTC"


# ---------------------------------------------------------------------------
# fetch_articles_from_seed (offline; no network)
# ---------------------------------------------------------------------------


def test_fetch_articles_from_seed_returns_utc_aware_frame():
    df = fetch_articles_from_seed(SEED_PATH)
    assert list(df.columns) == CANONICAL_ARTICLE_COLUMNS
    assert df["published_at"].dt.tz is not None
    assert str(df["published_at"].dt.tz) == "UTC"
    assert len(df) >= 10  # seed has 10 rows
    # Tickers from the seed round-trip unchanged.
    assert "AAPL" in set(df["ticker"])
    assert "ENRN" in set(df["ticker"])  # not in universe, but seed loads it


# ---------------------------------------------------------------------------
# (e)-(g): join_articles_to_prices
# ---------------------------------------------------------------------------


def _build_synthetic_prices() -> pd.DataFrame:
    """Build a tiny synthetic prices frame with tz-aware America/New_York
    session closes matching the XNYS schedule for 2024-07-01..2024-07-08.

    Sessions (from a calendar smoke): 2024-07-01 (16:00), 07-02 (16:00),
    07-03 (13:00 early close), 07-05 (16:00), 07-08 (16:00). 07-04 (holiday)
    and 07-06/07-07 (weekend) are absent.
    """
    sessions = [
        pd.Timestamp("2024-07-01 16:00", tz="America/New_York"),
        pd.Timestamp("2024-07-02 16:00", tz="America/New_York"),
        pd.Timestamp("2024-07-03 13:00", tz="America/New_York"),
        pd.Timestamp("2024-07-05 16:00", tz="America/New_York"),
        pd.Timestamp("2024-07-08 16:00", tz="America/New_York"),
    ]
    rows = []
    for tk in ("AAPL", "MSFT", "NVDA"):
        for ts in sessions:
            rows.append(
                {
                    "ticker": tk,
                    "period_close_ts": ts,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "adj_close": 100.5,
                    "volume": 1_000_000.0,
                    "vwap": 100.0,
                }
            )
    return pd.DataFrame(rows)


def _build_join_loader(tmp_path: Path, prices_df: pd.DataFrame) -> EquityDataLoader:
    """Build a loader whose universe file includes AAPL/MSFT/NVDA alive for
    the 2024-07 window, write prices + articles stores to tmp_path.
    """
    prices_dir = tmp_path / "prices"
    articles_dir = tmp_path / "articles"
    prices_dir.mkdir()
    articles_dir.mkdir()
    # Write a single-file parquet (not partitioned) for the synthetic prices;
    # the loader reads it back and selects the 9 canonical columns.
    prices_path = prices_dir / "prices.parquet"
    prices_df.to_parquet(prices_path, index=False)

    # Build articles covering normal / holiday / weekend cases (UTC).
    articles = pd.DataFrame(
        {
            "ticker": [
                "AAPL",
                "MSFT",
                "AAPL",
                "MSFT",  # holiday 07-04 -> binds 07-05
                "AAPL",
                "NVDA",  # weekend 07-06 -> binds 07-08
                "AAPL",
                "MSFT",  # Monday 07-08
            ],
            "published_at": pd.to_datetime(
                [
                    "2024-07-01T14:30:00+00:00",
                    "2024-07-02T09:15:00+00:00",
                    "2024-07-03T17:00:00+00:00",
                    "2024-07-04T16:00:00+00:00",
                    "2024-07-05T11:00:00+00:00",
                    "2024-07-06T13:00:00+00:00",
                    "2024-07-08T10:00:00+00:00",
                    "2024-07-08T15:00:00+00:00",
                ],
                utc=True,
            ),
            "text": [f"text {i}" for i in range(8)],
            "source": [
                "reuters",
                "bloomberg",
                "reuters",
                "bloomberg",
                "reuters",
                "bloomberg",
                "reuters",
                "bloomberg",
            ],
        }
    )
    articles_path = articles_dir / "articles.parquet"
    articles.to_parquet(articles_path, index=False)

    loader = EquityDataLoader("sp500", "2024-07-01", "2024-07-08")
    return loader, prices_path, articles_path


def test_join_assigns_every_article_to_exactly_one_period(tmp_path):
    prices = _build_synthetic_prices()
    loader, prices_path, articles_path = _build_join_loader(tmp_path, prices)
    out = loader.join_articles_to_prices(
        prices_path=prices_path, articles_path=articles_path
    )
    joined = pd.read_parquet(out)
    # Each article binds to exactly one period: no duplicate (ticker, published_at, source).
    assert not joined.duplicated(
        subset=["ticker", "published_at", "source"]
    ).any()
    assert list(joined.columns) == [
        "ticker",
        "period_close_ts",
        "published_at",
        "text",
        "source",
    ]
    assert joined["period_close_ts"].dt.tz is not None
    assert str(joined["period_close_ts"].dt.tz) == "America/New_York"
    assert joined["published_at"].dt.tz is not None
    assert str(joined["published_at"].dt.tz) == "UTC"

    # The PIT inequality must hold for every row:
    #   period_close(P-1) < published_at <= period_close(P)
    pc_utc = joined["period_close_ts"].dt.tz_convert("UTC")
    pub = joined["published_at"].dt.tz_convert("UTC")
    assert (pub <= pc_utc).all(), "right-hand inequality violated"
    # Build the sorted session-close index per row and check the left-hand side.
    closes = pd.DatetimeIndex(
        sorted(set(joined["period_close_ts"].dt.tz_convert("UTC")))
    )
    for _, row in joined.iterrows():
        p = pd.Timestamp(row["published_at"]).tz_convert("UTC")
        c = pd.Timestamp(row["period_close_ts"]).tz_convert("UTC")
        idx = closes.get_loc(c)
        if idx > 0:
            assert closes[idx - 1] < p <= closes[idx]


def test_holiday_article_rolls_to_next_trading_period(tmp_path):
    # 2024-07-04 is a US holiday; an article that day must bind to 2024-07-05.
    prices = _build_synthetic_prices()
    loader, prices_path, articles_path = _build_join_loader(tmp_path, prices)
    joined = pd.read_parquet(
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )
    )
    holiday = joined[
        joined["published_at"]
        == pd.Timestamp("2024-07-04T16:00:00", tz="UTC")
    ].iloc[0]
    assert pd.Timestamp(holiday["period_close_ts"]).date() == pd.Timestamp(
        "2024-07-05"
    ).date()


def test_weekend_article_rolls_to_monday_period(tmp_path):
    # 2024-07-06 is a Saturday; an article that day must bind to 2024-07-08.
    prices = _build_synthetic_prices()
    loader, prices_path, articles_path = _build_join_loader(tmp_path, prices)
    joined = pd.read_parquet(
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )
    )
    weekend = joined[
        joined["published_at"]
        == pd.Timestamp("2024-07-06T13:00:00", tz="UTC")
    ].iloc[0]
    assert pd.Timestamp(weekend["period_close_ts"]).date() == pd.Timestamp(
        "2024-07-08"
    ).date()


def test_join_drops_articles_for_out_of_window_tickers(tmp_path):
    # ENRN (not in sp500 universe) and LEH (delisted 2008, out of a 2024
    # window) must be filtered out by the alive-ticker check.
    prices = _build_synthetic_prices()
    loader, prices_path, articles_path = _build_join_loader(tmp_path, prices)
    # Append out-of-window ticker rows to the articles store and rewrite.
    base = pd.read_parquet(articles_path)
    extra = pd.DataFrame(
        {
            "ticker": ["ENRN", "LEH"],
            "published_at": pd.to_datetime(
                ["2024-07-02T10:00:00+00:00", "2024-07-02T11:00:00+00:00"],
                utc=True,
            ),
            "text": ["enron news", "lehman news"],
            "source": ["reuters", "bloomberg"],
        }
    )
    pd.concat([base, extra], ignore_index=True).to_parquet(articles_path, index=False)
    joined = pd.read_parquet(
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )
    )
    assert "ENRN" not in set(joined["ticker"])
    assert "LEH" not in set(joined["ticker"])


# ---------------------------------------------------------------------------
# (h)-(j): published_at_guard
# ---------------------------------------------------------------------------


def test_run_published_at_guard_passes_on_correctly_joined_frame(tmp_path):
    prices = _build_synthetic_prices()
    loader, prices_path, articles_path = _build_join_loader(tmp_path, prices)
    joined = pd.read_parquet(
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )
    )
    result = run_published_at_guard(joined)
    assert result["pass"] is True
    assert result["n_violations"] == 0
    assert result["n_checked"] == len(joined)


def test_run_published_at_guard_fails_on_injected_violation(tmp_path):
    prices = _build_synthetic_prices()
    loader, prices_path, articles_path = _build_join_loader(tmp_path, prices)
    joined = pd.read_parquet(
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )
    )
    # Inject a violation: bump one row's published_at PAST its period close.
    bad = joined.copy()
    pc = pd.Timestamp(bad.iloc[0]["period_close_ts"]).tz_convert("UTC")
    bad.loc[bad.index[0], "published_at"] = pc + pd.Timedelta(hours=2)
    result = run_published_at_guard(bad)
    assert result["pass"] is False
    assert result["n_violations"] >= 1
    assert result["n_checked"] == len(bad)
    assert result["violations"][0]["gap_seconds"] > 0


def test_published_at_guard_cli_exits_nonzero_on_violation(tmp_path):
    prices = _build_synthetic_prices()
    loader, prices_path, articles_path = _build_join_loader(tmp_path, prices)
    joined = pd.read_parquet(
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )
    )
    # Inject a violation and write to a pre-joined parquet for the CLI.
    bad = joined.copy()
    pc = pd.Timestamp(bad.iloc[0]["period_close_ts"]).tz_convert("UTC")
    bad.loc[bad.index[0], "published_at"] = pc + pd.Timedelta(hours=2)
    joined_path = tmp_path / "articles_joined_bad.parquet"
    bad.to_parquet(joined_path, index=False)

    out_path = tmp_path / "guard_result.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "equity.diagnostics.published_at_guard",
            "--joined",
            str(joined_path),
            "--output",
            str(out_path),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, f"expected exit 1, got {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    assert out_path.exists()
    import json

    payload = json.loads(out_path.read_text())
    assert payload["pass"] is False
    assert payload["n_violations"] >= 1


def test_published_at_guard_cli_exits_zero_on_clean_joined(tmp_path):
    prices = _build_synthetic_prices()
    loader, prices_path, articles_path = _build_join_loader(tmp_path, prices)
    joined = pd.read_parquet(
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )
    )
    joined_path = tmp_path / "articles_joined_clean.parquet"
    joined.to_parquet(joined_path, index=False)
    out_path = tmp_path / "guard_clean.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "equity.diagnostics.published_at_guard",
            "--joined",
            str(joined_path),
            "--output",
            str(out_path),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"expected exit 0, got {proc.returncode}\n{proc.stdout}\n{proc.stderr}"
