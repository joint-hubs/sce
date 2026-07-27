"""
@module: tests.equity.test_sentiment_aggregate
@depends: equity.sentiment.aggregate, equity.sentiment.schema
@exports:
@data_flow: hand-built per-article fixture -> aggregate_per_period /
            aggregate_market_wide -> numeric formula to 1e-9
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from equity.sentiment.aggregate import (
    DEFAULT_HALFLIFE_DAYS,
    aggregate_market_wide,
    aggregate_per_period,
    build_market_wide_per_article,
    write_market_wide,
    write_sentiment_per_period,
)
from equity.sentiment.schema import (
    CANONICAL_MARKET_WIDE_COLUMNS,
    CANONICAL_PER_PERIOD_COLUMNS,
)


def _per_article_fixture() -> pd.DataFrame:
    """Hand-built fixture: 1 ticker, 1 period, 3 articles at known offsets.

    period_close = 2024-07-08 20:00 UTC (16:00 ET).
    Articles published at:
      a1: 2024-07-08 18:00 UTC  (dt = 2h  = 2/24 d)
      a2: 2024-07-08 12:00 UTC  (dt = 8h  = 8/24 d)
      a3: 2024-07-05 12:00 UTC  (dt = 3d + 8h = 3.333... d)
    """
    period_close = pd.Timestamp("2024-07-08 20:00", tz="UTC")
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "AAPL"],
            "published_at": [
                pd.Timestamp("2024-07-08 18:00", tz="UTC"),
                pd.Timestamp("2024-07-08 12:00", tz="UTC"),
                pd.Timestamp("2024-07-05 12:00", tz="UTC"),
            ],
            "period_close_ts": [period_close] * 3,
            "pos": [0.7, 0.5, 0.6],
            "neg": [0.2, 0.3, 0.1],
            "neu": [0.1, 0.2, 0.3],
            "score": [0.5, 0.2, 0.5],
        }
    )


def _expected_weighted_mean(values: list[float], dts_days: list[float], halflife: float) -> float:
    weights = [math.exp(-dt / halflife) for dt in dts_days]
    num = sum(w * v for w, v in zip(weights, values))
    den = sum(weights)
    return num / den


def test_aggregate_per_period_formula_to_1e9():
    df = _per_article_fixture()
    out = aggregate_per_period(df, halflife_days=DEFAULT_HALFLIFE_DAYS)
    assert len(out) == 1
    row = out.iloc[0]
    dts_days = [2.0 / 24.0, 8.0 / 24.0, 3.0 + 8.0 / 24.0]
    assert row["sentiment_score"] == pytest.approx(
        _expected_weighted_mean([0.5, 0.2, 0.5], dts_days, DEFAULT_HALFLIFE_DAYS),
        abs=1e-9,
    )
    assert row["sentiment_pos"] == pytest.approx(
        _expected_weighted_mean([0.7, 0.5, 0.6], dts_days, DEFAULT_HALFLIFE_DAYS),
        abs=1e-9,
    )
    assert row["sentiment_neg"] == pytest.approx(
        _expected_weighted_mean([0.2, 0.3, 0.1], dts_days, DEFAULT_HALFLIFE_DAYS),
        abs=1e-9,
    )
    assert row["sentiment_neu"] == pytest.approx(
        _expected_weighted_mean([0.1, 0.2, 0.3], dts_days, DEFAULT_HALFLIFE_DAYS),
        abs=1e-9,
    )
    assert row["n_articles"] == 3


def test_aggregate_per_period_pos_neg_neu_sum_to_one():
    df = _per_article_fixture()
    out = aggregate_per_period(df)
    s = out["sentiment_pos"] + out["sentiment_neg"] + out["sentiment_neu"]
    assert (s - 1.0).abs().max() < 1e-9


def test_aggregate_per_period_weight_monotonicity():
    """Older articles (smaller published_at) receive smaller-or-equal
    weights. Indirectly asserted via the weighted-mean formula: replacing
    the oldest article's score with a very high value must pull the
    weighted mean UP by less than replacing the newest article's score.
    """
    df = _per_article_fixture()
    base = aggregate_per_period(df).iloc[0]["sentiment_score"]

    # Bump the NEWEST article's score by +0.3.
    df_new = df.copy()
    df_new.loc[0, "score"] = 0.8
    out_new = aggregate_per_period(df_new).iloc[0]["sentiment_score"]

    # Bump the OLDEST article's score by +0.3.
    df_old = df.copy()
    df_old.loc[2, "score"] = 0.8
    out_old = aggregate_per_period(df_old).iloc[0]["sentiment_score"]

    # Newest has more weight, so the same +0.3 bump moves the mean more.
    assert (out_new - base) > (out_old - base)


def test_aggregate_per_period_multiple_tickers():
    period_close = pd.Timestamp("2024-07-08 20:00", tz="UTC")
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "published_at": [
                pd.Timestamp("2024-07-08 18:00", tz="UTC"),
                pd.Timestamp("2024-07-08 12:00", tz="UTC"),
                pd.Timestamp("2024-07-08 18:00", tz="UTC"),
                pd.Timestamp("2024-07-08 12:00", tz="UTC"),
            ],
            "period_close_ts": [period_close] * 4,
            "pos": [0.7, 0.5, 0.6, 0.4],
            "neg": [0.2, 0.3, 0.3, 0.5],
            "neu": [0.1, 0.2, 0.1, 0.1],
            "score": [0.5, 0.2, 0.3, -0.1],
        }
    )
    out = aggregate_per_period(df)
    assert len(out) == 2
    assert set(out["ticker"]) == {"AAPL", "MSFT"}
    assert list(out.columns) == CANONICAL_PER_PERIOD_COLUMNS


def test_aggregate_per_period_rejects_pit_violation():
    """An article with published_at > period_close_ts must raise."""
    period_close = pd.Timestamp("2024-07-08 20:00", tz="UTC")
    df = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "published_at": [pd.Timestamp("2024-07-08 21:00", tz="UTC")],
            "period_close_ts": [period_close],
            "pos": [0.7],
            "neg": [0.2],
            "neu": [0.1],
            "score": [0.5],
        }
    )
    with pytest.raises(ValueError, match="PIT violation"):
        aggregate_per_period(df)


def test_aggregate_per_period_rejects_null_ticker():
    period_close = pd.Timestamp("2024-07-08 20:00", tz="UTC")
    df = pd.DataFrame(
        {
            "ticker": [None],
            "published_at": [pd.Timestamp("2024-07-08 18:00", tz="UTC")],
            "period_close_ts": [period_close],
            "pos": [0.7],
            "neg": [0.2],
            "neu": [0.1],
            "score": [0.5],
        }
    )
    with pytest.raises(ValueError, match="null ticker"):
        aggregate_per_period(df)


def test_aggregate_per_period_empty_returns_empty_canonical():
    out = aggregate_per_period(
        pd.DataFrame(
            columns=["ticker", "published_at", "period_close_ts", "pos", "neg", "neu", "score"]
        )
    )
    assert list(out.columns) == CANONICAL_PER_PERIOD_COLUMNS
    assert len(out) == 0


def test_aggregate_market_wide_formula():
    period_close = pd.Timestamp("2024-07-08 20:00", tz="UTC")
    df = pd.DataFrame(
        {
            "ticker": ["__SENTINEL__", "__OTHER__"],
            "published_at": [
                pd.Timestamp("2024-07-08 18:00", tz="UTC"),
                pd.Timestamp("2024-07-08 12:00", tz="UTC"),
            ],
            "period_close_ts": [period_close, period_close],
            "pos": [0.7, 0.5],
            "neg": [0.2, 0.3],
            "neu": [0.1, 0.2],
            "score": [0.5, 0.2],
        }
    )
    out = aggregate_market_wide(df, halflife_days=DEFAULT_HALFLIFE_DAYS)
    assert len(out) == 1
    row = out.iloc[0]
    dts_days = [2.0 / 24.0, 8.0 / 24.0]
    assert row["sentiment_score"] == pytest.approx(
        _expected_weighted_mean([0.5, 0.2], dts_days, DEFAULT_HALFLIFE_DAYS),
        abs=1e-9,
    )
    assert row["n_articles"] == 2
    assert "ticker" not in out.columns
    assert list(out.columns) == CANONICAL_MARKET_WIDE_COLUMNS


def test_aggregate_market_wide_rejects_pit_violation():
    period_close = pd.Timestamp("2024-07-08 20:00", tz="UTC")
    df = pd.DataFrame(
        {
            "ticker": ["X"],
            "published_at": [pd.Timestamp("2024-07-08 21:00", tz="UTC")],
            "period_close_ts": [period_close],
            "pos": [0.7],
            "neg": [0.2],
            "neu": [0.1],
            "score": [0.5],
        }
    )
    with pytest.raises(ValueError, match="PIT violation"):
        aggregate_market_wide(df)


def test_build_market_wide_per_article_assigns_xnys_close():
    """An article published 2024-07-09 (Tuesday) 18:00 UTC binds to the
    2024-07-09 XNYS close (20:00 UTC). A Saturday article published
    2024-07-06 18:00 UTC binds to the NEXT session close (2024-07-08
    Monday 20:00 UTC) -- pre-first-session articles are NOT dropped in the
    market-wide path (no PIT training pair is being formed; the LHS of the
    PIT rule is vacuous for the first session).
    """
    articles = pd.DataFrame(
        {
            "ticker": ["A", "B"],
            "published_at": [
                pd.Timestamp("2024-07-09 18:00", tz="UTC"),  # Tuesday 18:00 UTC
                pd.Timestamp("2024-07-06 18:00", tz="UTC"),  # Saturday
            ],
            "text": ["tuesday article", "weekend article"],
            "source": ["reuters", "reuters"],
        }
    )
    out = build_market_wide_per_article(
        articles,
        window_start=pd.Timestamp("2024-07-08", tz="UTC"),
        window_end=pd.Timestamp("2024-07-10", tz="UTC"),
    )
    # Tuesday 18:00 UTC -> binds to 2024-07-09 20:00 UTC XNYS close.
    tue = out[out["ticker"] == "A"].iloc[0]
    assert tue["period_close_ts"] == pd.Timestamp("2024-07-09 20:00", tz="UTC")
    # Saturday article: 2024-07-06 -> binds to the NEXT session close
    # (Monday 2024-07-08 20:00 UTC), NOT dropped.
    sat = out[out["ticker"] == "B"].iloc[0]
    assert sat["period_close_ts"] == pd.Timestamp("2024-07-08 20:00", tz="UTC")


def test_build_market_wide_per_article_drops_post_window():
    """An article published after the last XNYS session in the window is
    dropped (under-inclusion, NOT leakage -- mirrors S1.3).
    """
    articles = pd.DataFrame(
        {
            "ticker": ["LATE"],
            "published_at": [pd.Timestamp("2024-07-15 18:00", tz="UTC")],
            "text": ["post-window article"],
            "source": ["reuters"],
        }
    )
    out = build_market_wide_per_article(
        articles,
        window_start=pd.Timestamp("2024-07-08", tz="UTC"),
        window_end=pd.Timestamp("2024-07-09", tz="UTC"),
    )
    # window_end + 2d = 2024-07-11. The 2024-07-15 article is after the last
    # session close (2024-07-11) -> bfill returns -1 -> dropped.
    assert len(out) == 0


def test_build_market_wide_per_article_empty_input():
    articles = pd.DataFrame(columns=["ticker", "published_at", "text", "source"])
    out = build_market_wide_per_article(articles)
    assert len(out) == 0
    assert "period_close_ts" in out.columns


def test_aggregate_per_period_halflife_infinity_converges_to_simple_mean():
    """As halflife -> inf, the weighted mean converges to the simple mean."""
    df = _per_article_fixture()
    # Use a very large halflife (effectively uniform weights).
    out_large = aggregate_per_period(df, halflife_days=1e6)
    simple_mean = df["score"].mean()
    assert out_large.iloc[0]["sentiment_score"] == pytest.approx(simple_mean, abs=1e-6)


def test_aggregate_per_period_halflife_zero_raises():
    df = _per_article_fixture()
    with pytest.raises(ValueError, match="halflife_days must be > 0"):
        aggregate_per_period(df, halflife_days=0)


def test_aggregate_per_period_missing_columns_raises():
    df = pd.DataFrame({"ticker": [], "published_at": []})
    with pytest.raises(ValueError, match="missing required columns"):
        aggregate_per_period(df)


# ---------------------------------------------------------------------------
# Writers (Hive-partitioned parquet)
# ---------------------------------------------------------------------------


def test_write_sentiment_per_period_writes_partitioned_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr("equity.sentiment.aggregate.PROJECT_ROOT", tmp_path)
    per_period = aggregate_per_period(_per_article_fixture())
    out_dir = write_sentiment_per_period(per_period, "data/equity/sentiment_per_period")
    assert out_dir.exists()
    # Hive partitions: year/month directories under out_dir.
    parts = list(out_dir.glob("year=*/month=*"))
    assert parts, f"no Hive partitions under {out_dir}"
    # Read-back: pandas reassembles the frame (drops partition keys).
    df = pd.read_parquet(out_dir)
    assert "ticker" in df.columns
    assert "period_close_ts" in df.columns
    assert "sentiment_score" in df.columns
    assert len(df) == 1


def test_write_market_wide_writes_partitioned_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr("equity.sentiment.aggregate.PROJECT_ROOT", tmp_path)
    period_close = pd.Timestamp("2024-07-08 20:00", tz="UTC")
    market_wide = aggregate_market_wide(
        pd.DataFrame(
            {
                "ticker": ["X"],
                "published_at": [pd.Timestamp("2024-07-08 18:00", tz="UTC")],
                "period_close_ts": [period_close],
                "pos": [0.7],
                "neg": [0.2],
                "neu": [0.1],
                "score": [0.5],
            }
        )
    )
    out_dir = write_market_wide(market_wide, "data/equity/market_wide_sentiment")
    assert out_dir.exists()
    parts = list(out_dir.glob("year=*/month=*"))
    assert parts, f"no Hive partitions under {out_dir}"
    df = pd.read_parquet(out_dir)
    assert "ticker" not in df.columns
    assert "period_close_ts" in df.columns
    assert len(df) == 1


def test_write_sentiment_per_period_rejects_outside_project_root(tmp_path):
    period_close = pd.Timestamp("2024-07-08 20:00", tz="UTC")
    df = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "period_close_ts": [period_close],
            "sentiment_score": [0.5],
            "sentiment_pos": [0.7],
            "sentiment_neg": [0.2],
            "sentiment_neu": [0.1],
            "n_articles": [1],
        }
    )
    with pytest.raises(ValueError, match="outside PROJECT_ROOT"):
        write_sentiment_per_period(df, str(tmp_path / "outside"))
