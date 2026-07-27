"""
@module: tests.equity.test_sentiment_schema
@depends: equity.sentiment.schema, pandera, pandas
@exports:
@data_flow: synthetic frames -> validate_* / assert_*_primary_key_unique
"""

from __future__ import annotations

import pandas as pd
import pytest

from equity.sentiment.schema import (
    CANONICAL_MARKET_WIDE_COLUMNS,
    CANONICAL_PER_ARTICLE_COLUMNS,
    CANONICAL_PER_PERIOD_COLUMNS,
    assert_market_wide_primary_key_unique,
    assert_per_article_primary_key_unique,
    assert_per_period_primary_key_unique,
    market_wide_schema,
    sentiment_per_article_schema,
    sentiment_per_period_schema,
    validate_market_wide,
    validate_sentiment_per_article,
    validate_sentiment_per_period,
)


def _valid_per_article(n: int = 2) -> pd.DataFrame:
    # Cache-internal canonical frame: NO ticker / published_at (those are
    # pass-through from the input at read time; see FOC-49 B2).
    return pd.DataFrame(
        {
            "article_key": [f"key{i}" for i in range(n)],
            "model_name": ["stub"] * n,
            "model_revision": ["v1"] * n,
            "pos": [0.5] * n,
            "neg": [0.3] * n,
            "neu": [0.2] * n,
            "score": [0.2] * n,
        }
    )


def _valid_per_period(n: int = 1) -> pd.DataFrame:
    pc = pd.date_range("2024-07-08", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ticker": ["AAPL"] * n,
            "period_close_ts": pc,
            "sentiment_score": [0.2] * n,
            "sentiment_pos": [0.5] * n,
            "sentiment_neg": [0.3] * n,
            "sentiment_neu": [0.2] * n,
            "n_articles": [1] * n,
        }
    )


def _valid_market_wide(n: int = 1) -> pd.DataFrame:
    pc = pd.date_range("2024-07-08", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "period_close_ts": pc,
            "sentiment_score": [0.2] * n,
            "sentiment_pos": [0.5] * n,
            "sentiment_neg": [0.3] * n,
            "sentiment_neu": [0.2] * n,
            "n_articles": [1] * n,
        }
    )


# ---------------------------------------------------------------------------
# per-article
# ---------------------------------------------------------------------------


def test_validate_per_article_passes_valid_frame():
    df = _valid_per_article()
    out = validate_sentiment_per_article(df)
    assert list(out.columns) == CANONICAL_PER_ARTICLE_COLUMNS


def test_validate_per_article_rejects_tz_naive_published_at():
    # No published_at in the cache schema anymore (FOC-49 B2); this test
    # now only confirms that adding an unexpected column is rejected by
    # the strict schema.
    df = _valid_per_article()
    df["published_at"] = pd.to_datetime(pd.Series(["2024-07-01"] * len(df)), utc=True)
    with pytest.raises(Exception):
        validate_sentiment_per_article(df)


def test_validate_per_article_rejects_prob_sum_violation():
    df = _valid_per_article()
    df.loc[0, "neu"] = 0.5  # 0.5 + 0.3 + 0.5 = 1.3
    with pytest.raises(ValueError, match="probabilities invariant"):
        validate_sentiment_per_article(df)


def test_validate_per_article_rejects_nan_pos():
    """FOC-49 round-3: NaN in a probability column must be rejected at
    schema validation (nullable=False), not silently propagate downstream.
    """
    import numpy as np

    df = _valid_per_article()
    df.loc[0, "pos"] = np.nan
    with pytest.raises(Exception):
        validate_sentiment_per_article(df)


def test_validate_per_period_rejects_nan_sentiment_pos():
    """FOC-49 round-3: NaN in sentiment_pos must be rejected at schema
    validation (nullable=False) -- a NaN-producing scorer would otherwise
    bypass the prob-sum comparison guard (NaN comparisons are always False).
    """
    import numpy as np

    df = _valid_per_period()
    df.loc[0, "sentiment_pos"] = np.nan
    with pytest.raises(Exception):
        validate_sentiment_per_period(df)


def test_validate_market_wide_rejects_nan_sentiment_score():
    """FOC-49 round-3: NaN in sentiment_score must be rejected at schema
    validation (nullable=False)."""
    import numpy as np

    df = _valid_market_wide()
    df.loc[0, "sentiment_score"] = np.nan
    with pytest.raises(Exception):
        validate_market_wide(df)


def test_validate_per_article_empty_passes():
    empty = pd.DataFrame(
        {
            "article_key": pd.Series(dtype=str),
            "model_name": pd.Series(dtype=str),
            "model_revision": pd.Series(dtype=str),
            "pos": pd.Series(dtype=float),
            "neg": pd.Series(dtype=float),
            "neu": pd.Series(dtype=float),
            "score": pd.Series(dtype=float),
        }
    )
    out = validate_sentiment_per_article(empty)
    assert len(out) == 0


def test_assert_per_article_pk_unique_detects_duplicates():
    df = _valid_per_article(n=2)
    df.loc[1, "article_key"] = "key0"  # duplicate
    with pytest.raises(ValueError, match="Primary key"):
        assert_per_article_primary_key_unique(df)


def test_assert_per_article_pk_unique_noop_on_empty():
    assert_per_article_primary_key_unique(pd.DataFrame(columns=CANONICAL_PER_ARTICLE_COLUMNS))


def test_per_article_schema_is_strict():
    assert sentiment_per_article_schema.strict is True


# ---------------------------------------------------------------------------
# per-period
# ---------------------------------------------------------------------------


def test_validate_per_period_passes_valid_frame():
    df = _valid_per_period()
    out = validate_sentiment_per_period(df)
    assert list(out.columns) == CANONICAL_PER_PERIOD_COLUMNS


def test_validate_per_period_rejects_tz_naive_period_close():
    df = _valid_per_period()
    df["period_close_ts"] = df["period_close_ts"].dt.tz_localize(None)
    with pytest.raises(Exception):
        validate_sentiment_per_period(df)


def test_validate_per_period_rejects_prob_sum_violation():
    df = _valid_per_period()
    df.loc[0, "sentiment_neg"] = 0.9  # 0.5 + 0.9 + 0.2 = 1.6
    with pytest.raises(ValueError, match="probabilities invariant"):
        validate_sentiment_per_period(df)


def test_validate_per_period_rejects_nan_prob_sum_via_custom_check():
    """FOC-49 round-3: even if a NaN slipped past the nullable=False schema
    check (e.g. via a frame constructed with float dtype that pandera
    coerces), the custom prob-sum check must flag NaN as a violation
    (NaN comparisons are always False -- the guard adds np.isnan to bad_mask).
    Here we test the schema path with a non-NaN prob-sum violation already
    covered above; this test confirms the NaN path of the custom check via
    a direct call to _assert_probs_sum_to_one."""
    import numpy as np

    from equity.sentiment.schema import _assert_probs_sum_to_one

    df = _valid_per_period()
    df.loc[0, "sentiment_pos"] = np.nan
    # Force the row past the nullable=False schema guard by calling the
    # custom check directly (the schema would already reject this).
    with pytest.raises(ValueError, match="probabilities invariant"):
        _assert_probs_sum_to_one(df, "sentiment_pos", "sentiment_neg", "sentiment_neu")


def test_assert_per_period_pk_unique_detects_duplicates():
    df = _valid_per_period(n=2)
    df.loc[1, "period_close_ts"] = df.loc[0, "period_close_ts"]
    with pytest.raises(ValueError, match="Primary key"):
        assert_per_period_primary_key_unique(df)


def test_assert_per_period_pk_unique_noop_on_empty():
    assert_per_period_primary_key_unique(pd.DataFrame(columns=CANONICAL_PER_PERIOD_COLUMNS))


def test_per_period_schema_is_strict():
    assert sentiment_per_period_schema.strict is True


# ---------------------------------------------------------------------------
# market-wide
# ---------------------------------------------------------------------------


def test_validate_market_wide_passes_valid_frame():
    df = _valid_market_wide()
    out = validate_market_wide(df)
    assert list(out.columns) == CANONICAL_MARKET_WIDE_COLUMNS


def test_validate_market_wide_rejects_tz_naive_period_close():
    df = _valid_market_wide()
    df["period_close_ts"] = df["period_close_ts"].dt.tz_localize(None)
    with pytest.raises(Exception):
        validate_market_wide(df)


def test_validate_market_wide_rejects_prob_sum_violation():
    df = _valid_market_wide()
    df.loc[0, "sentiment_neu"] = 0.8  # 0.5 + 0.3 + 0.8 = 1.6
    with pytest.raises(ValueError, match="probabilities invariant"):
        validate_market_wide(df)


def test_assert_market_wide_pk_unique_detects_duplicates():
    df = _valid_market_wide(n=2)
    df.loc[1, "period_close_ts"] = df.loc[0, "period_close_ts"]
    with pytest.raises(ValueError, match="Primary key"):
        assert_market_wide_primary_key_unique(df)


def test_assert_market_wide_pk_unique_noop_on_empty():
    assert_market_wide_primary_key_unique(pd.DataFrame(columns=CANONICAL_MARKET_WIDE_COLUMNS))


def test_market_wide_schema_is_strict():
    assert market_wide_schema.strict is True
