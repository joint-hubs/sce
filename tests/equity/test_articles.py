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

import pandas as pd
import pytest

from equity.data.fetch import fetch_articles_from_seed
from equity.data.loader import EquityDataLoader
from equity.data.registry import PROJECT_ROOT
from equity.data.schema import (
    ARTICLE_TZ,
    CANONICAL_ARTICLE_COLUMNS,
    articles_schema,
    assert_articles_primary_key_unique,
    validate_articles,
)
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
    # The synthetic out-of-universe sentinel (m4) is loaded by the seed.
    assert "__TEST_NOT_IN_UNIVERSE__" in set(df["ticker"])


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
                    "hlc_average": 100.0,
                }
            )
    return pd.DataFrame(rows)


def _build_join_loader(
    tmp_path: Path, prices_df: pd.DataFrame
) -> tuple[EquityDataLoader, Path, Path]:
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
    # n3: both columns canonicalized to UTC in the joined output.
    assert str(joined["period_close_ts"].dt.tz) == "UTC"
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
    # The synthetic __TEST_NOT_IN_UNIVERSE__ sentinel (not in sp500 universe)
    # and LEH (delisted 2008, out of a 2024 window) must be filtered out by
    # the alive-ticker check.
    prices = _build_synthetic_prices()
    loader, prices_path, articles_path = _build_join_loader(tmp_path, prices)
    # Append out-of-window ticker rows to the articles store and rewrite.
    base = pd.read_parquet(articles_path)
    extra = pd.DataFrame(
        {
            "ticker": ["__TEST_NOT_IN_UNIVERSE__", "LEH"],
            "published_at": pd.to_datetime(
                ["2024-07-02T10:00:00+00:00", "2024-07-02T11:00:00+00:00"],
                utc=True,
            ),
            "text": ["sentinel news", "lehman news"],
            "source": ["reuters", "bloomberg"],
        }
    )
    pd.concat([base, extra], ignore_index=True).to_parquet(articles_path, index=False)
    joined = pd.read_parquet(
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )
    )
    assert "__TEST_NOT_IN_UNIVERSE__" not in set(joined["ticker"])
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


# ---------------------------------------------------------------------------
# m7: error-path / edge-case tests for the join + guard + seed loader.
# ---------------------------------------------------------------------------


def test_fetch_articles_from_seed_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="not found"):
        fetch_articles_from_seed(missing)


def test_fetch_articles_from_seed_missing_column_raises(tmp_path):
    bad = tmp_path / "bad_seed.csv"
    bad.write_text(
        "ticker,published_at,text\n"  # missing 'source'
        "AAPL,2024-07-01T10:00:00+00:00,hello\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing required column 'source'"):
        fetch_articles_from_seed(bad)


def test_fetch_articles_from_seed_empty_returns_empty_frame(tmp_path):
    empty = tmp_path / "empty_seed.csv"
    empty.write_text(
        "# only a comment line\nticker,published_at,text,source\n",
        encoding="utf-8",
    )
    out = fetch_articles_from_seed(empty)
    assert out.empty
    assert list(out.columns) == CANONICAL_ARTICLE_COLUMNS


def test_join_missing_prices_store_raises_file_not_found(tmp_path):
    articles = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "published_at": pd.to_datetime(["2024-07-01T10:00:00+00:00"], utc=True),
            "text": ["x"],
            "source": ["r"],
        }
    )
    articles_path = tmp_path / "articles.parquet"
    articles.to_parquet(articles_path, index=False)
    loader = EquityDataLoader("sp500", "2024-07-01", "2024-07-08")
    with pytest.raises(FileNotFoundError, match="Prices store not found"):
        loader.join_articles_to_prices(
            prices_path=tmp_path / "missing_prices.parquet",
            articles_path=articles_path,
        )


def test_join_missing_articles_store_raises_file_not_found(tmp_path):
    prices = _build_synthetic_prices()
    prices_path = tmp_path / "prices.parquet"
    prices.to_parquet(prices_path, index=False)
    loader = EquityDataLoader("sp500", "2024-07-01", "2024-07-08")
    with pytest.raises(FileNotFoundError, match="Articles store not found"):
        loader.join_articles_to_prices(
            prices_path=prices_path,
            articles_path=tmp_path / "missing_articles.parquet",
        )


def test_join_prices_missing_canonical_column_raises(tmp_path):
    prices = _build_synthetic_prices().drop(columns=["hlc_average"])
    prices_path = tmp_path / "prices.parquet"
    prices.to_parquet(prices_path, index=False)
    articles = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "published_at": pd.to_datetime(["2024-07-01T10:00:00+00:00"], utc=True),
            "text": ["x"],
            "source": ["r"],
        }
    )
    articles_path = tmp_path / "articles.parquet"
    articles.to_parquet(articles_path, index=False)
    loader = EquityDataLoader("sp500", "2024-07-01", "2024-07-08")
    with pytest.raises(ValueError, match="Prices store missing canonical columns"):
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )


def test_join_empty_articles_after_alive_filter_raises(tmp_path):
    prices = _build_synthetic_prices()
    prices_path = tmp_path / "prices.parquet"
    prices.to_parquet(prices_path, index=False)
    # Articles with ONLY out-of-window tickers -> empty after alive-filter.
    articles = pd.DataFrame(
        {
            "ticker": ["__TEST_NOT_IN_UNIVERSE__", "LEH"],
            "published_at": pd.to_datetime(
                ["2024-07-02T10:00:00+00:00", "2024-07-02T11:00:00+00:00"], utc=True
            ),
            "text": ["a", "b"],
            "source": ["r", "b"],
        }
    )
    articles_path = tmp_path / "articles.parquet"
    articles.to_parquet(articles_path, index=False)
    loader = EquityDataLoader("sp500", "2024-07-01", "2024-07-08")
    with pytest.raises(ValueError, match="empty after the alive-ticker filter"):
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )


def test_join_all_articles_after_last_session_raises(tmp_path):
    """All articles published AFTER the last stored session close -> no
    binding possible -> ValueError (under-inclusion at the window edge)."""
    prices = _build_synthetic_prices()  # last session = 2024-07-08 16:00 ET
    prices_path = tmp_path / "prices.parquet"
    prices.to_parquet(prices_path, index=False)
    articles = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            # 2024-07-09 10:00 UTC -> after the 2024-07-08 20:00 UTC close.
            "published_at": pd.to_datetime(
                ["2024-07-09T10:00:00+00:00"], utc=True
            ),
            "text": ["future"],
            "source": ["r"],
        }
    )
    articles_path = tmp_path / "articles.parquet"
    articles.to_parquet(articles_path, index=False)
    loader = EquityDataLoader("sp500", "2024-07-01", "2024-07-08")
    with pytest.raises(ValueError, match="No articles could be bound"):
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )


def test_run_published_at_guard_tz_naive_period_close_raises():
    joined = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "period_close_ts": pd.to_datetime(["2024-07-01 20:00"]),  # naive
            "published_at": pd.to_datetime(["2024-07-01T10:00:00+00:00"], utc=True),
            "text": ["x"],
            "source": ["r"],
        }
    )
    with pytest.raises(ValueError, match="period_close_ts is tz-naive"):
        run_published_at_guard(joined)


def test_run_published_at_guard_tz_naive_published_at_raises():
    joined = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "period_close_ts": pd.to_datetime(
                ["2024-07-01 20:00"], utc=True  # tz-aware UTC
            ),
            "published_at": pd.to_datetime(["2024-07-01 10:00"]),  # naive
            "text": ["x"],
            "source": ["r"],
        }
    )
    with pytest.raises(ValueError, match="published_at is tz-naive"):
        run_published_at_guard(joined)


def test_run_published_at_guard_missing_columns_raises():
    joined = pd.DataFrame({"ticker": ["AAPL"], "published_at": [pd.Timestamp.now(tz="UTC")]})
    with pytest.raises(ValueError, match="missing required columns"):
        run_published_at_guard(joined)


def test_run_published_at_guard_empty_frame_passes():
    empty = pd.DataFrame(
        columns=["ticker", "period_close_ts", "published_at", "text", "source"]
    )
    result = run_published_at_guard(empty)
    assert result["pass"] is True
    assert result["n_checked"] == 0


def test_run_published_at_guard_lhs_violation_detected():
    """Left-hand side: an article whose published_at is BEFORE the previous
    stored session close (but <= its own period close) -- e.g. an ffill
    regression. The LHS check must catch it (the RHS check alone would miss
    this because published_at <= period_close)."""
    # Two sessions in the joined frame: 2024-07-05 20:00, 2024-07-08 20:00 UTC.
    # Row 0 (first session): pub 2024-07-04 10:00 -> no previous session -> LHS
    # holds by definition; RHS holds (pub <= pc).
    # Row 1 (second session): pub 2024-07-04 10:00 -> RHS holds (pub <= pc), but
    # the previous session close (2024-07-05 20:00) is >= pub -> LHS violated.
    pc = pd.to_datetime(
        ["2024-07-05 20:00", "2024-07-08 20:00"], utc=True
    )
    joined = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "period_close_ts": pc,
            "published_at": pd.to_datetime(
                ["2024-07-04T10:00:00+00:00", "2024-07-04T10:00:00+00:00"], utc=True
            ),
            "text": ["first ok", "lhs leak"],
            "source": ["r", "b"],
        }
    )
    result = run_published_at_guard(joined)
    assert result["pass"] is False, f"expected LHS violation, got {result}"
    assert result["n_violations"] >= 1


# ---------------------------------------------------------------------------
# B2.3: FK integrity assertion (joined period_close_ts is a subset of prices).
# ---------------------------------------------------------------------------


def test_join_period_close_ts_is_valid_fk_into_prices(tmp_path):
    """The joined ``period_close_ts`` set must be a subset of the prices
    store's ``period_close_ts`` set (B2: no phantom FK)."""
    prices = _build_synthetic_prices()
    loader, prices_path, articles_path = _build_join_loader(tmp_path, prices)
    joined = pd.read_parquet(
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )
    )
    prices_ts_utc = set(prices["period_close_ts"].dt.tz_convert("UTC").unique())
    joined_ts_utc = set(joined["period_close_ts"].dt.tz_convert("UTC").unique())
    assert joined_ts_utc.issubset(prices_ts_utc), (
        f"FK violated: {joined_ts_utc - prices_ts_utc} not in prices"
    )


def test_assert_fk_integrity_passes_on_valid_join(tmp_path):
    from equity.diagnostics.published_at_guard import assert_fk_integrity

    prices = _build_synthetic_prices()
    loader, prices_path, articles_path = _build_join_loader(tmp_path, prices)
    joined = pd.read_parquet(
        loader.join_articles_to_prices(
            prices_path=prices_path, articles_path=articles_path
        )
    )
    # No exception -> FK holds.
    assert_fk_integrity(joined, prices)


def test_assert_fk_integrity_raises_on_phantom_key():
    from equity.diagnostics.published_at_guard import assert_fk_integrity

    prices = pd.DataFrame(
        {
            "period_close_ts": pd.to_datetime(
                ["2024-07-01 20:00", "2024-07-02 20:00"], utc=True
            ),
            "ticker": ["AAPL", "AAPL"],
        }
    )
    # Joined frame references a session NOT in prices (phantom FK).
    joined = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "period_close_ts": pd.to_datetime(["2024-07-08 20:00"], utc=True),
            "published_at": pd.to_datetime(["2024-07-01T10:00:00+00:00"], utc=True),
            "text": ["x"],
            "source": ["r"],
        }
    )
    with pytest.raises(ValueError, match="FK integrity violated"):
        assert_fk_integrity(joined, prices)


def test_published_at_guard_cli_runs_fk_check_when_prices_given(tmp_path):
    """When --joined AND --prices are both given, the CLI runs the FK check;
    a phantom key produces exit 1."""
    prices = _build_synthetic_prices()
    prices_path = tmp_path / "prices.parquet"
    prices.to_parquet(prices_path, index=False)

    # Build a joined frame with a phantom period_close_ts not in prices.
    joined = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "period_close_ts": pd.to_datetime(["2024-07-15 20:00"], utc=True),
            "published_at": pd.to_datetime(["2024-07-01T10:00:00+00:00"], utc=True),
            "text": ["phantom"],
            "source": ["r"],
        }
    )
    joined_path = tmp_path / "joined_phantom.parquet"
    joined.to_parquet(joined_path, index=False)
    out_path = tmp_path / "fk_result.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "equity.diagnostics.published_at_guard",
            "--joined",
            str(joined_path),
            "--prices",
            str(prices_path),
            "--output",
            str(out_path),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, (
        f"expected exit 1 on FK violation, got {proc.returncode}\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# M5: EquityDataLoader.fetch_articles() end-to-end.
# ---------------------------------------------------------------------------


def test_fetch_articles_writes_partitioned_parquet(tmp_path, monkeypatch):
    """EquityDataLoader.fetch_articles() end-to-end: loads the configured
    seed, validates, writes a partitioned parquet store, plus _meta.json."""
    # Point the sp500 config's seed_file + output_dir at tmp_path via a
    # monkeypatched CONFIG_DIR with a copy of the sp500 config + seed.
    import shutil as _sh

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    _sh.copy(PROJECT_ROOT / "configs" / "equity" / "sp500.toml", cfg_dir / "sp500.toml")
    _sh.copy(
        PROJECT_ROOT / "configs" / "equity" / "sp500_universe.csv",
        cfg_dir / "sp500_universe.csv",
    )
    seed = tmp_path / "seed.csv"
    seed.write_text(
        "ticker,published_at,text,source\n"
        "AAPL,2024-07-01T14:30:00+00:00,Apple news,reuters\n"
        "MSFT,2024-07-02T09:15:00+00:00,MSFT news,bloomberg\n",
        encoding="utf-8",
    )
    # Rewrite the copied TOML to point at the tmp seed + universe (absolute).
    (cfg_dir / "sp500.toml").write_text(
        '[universe]\n'
        'name = "sp500"\n'
        'description = "test"\n'
        'source = "test"\n'
        f"universe_file = '{(cfg_dir / 'sp500_universe.csv').as_posix()}'\n"
        '\n'
        '[prices]\n'
        'output_dir = "data/equity/prices"\n'
        'partition_cols = ["year", "month"]\n'
        'source = "yfinance"\n'
        '\n'
        '[articles]\n'
        f'output_dir = "{(tmp_path / "articles").as_posix()}"\n'
        'partition_cols = ["year", "month"]\n'
        'source = "seed"\n'
        f'seed_file = "{seed.as_posix()}"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", cfg_dir)

    loader = EquityDataLoader("sp500", "2024-07-01", "2024-07-31")
    out = loader.fetch_articles()  # uses [articles].output_dir from TOML
    assert out.exists()
    df = pd.read_parquet(out)
    assert {"ticker", "published_at", "text", "source"}.issubset(set(df.columns))
    assert len(df) == 2
    assert str(df["published_at"].dt.tz) == "UTC"
    # _meta.json provenance written.
    assert (out / "_meta.json").exists()

