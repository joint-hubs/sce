"""
@module: equity.sentiment.schema
@depends: pandera, pandas
@exports: sentiment_per_article_schema, sentiment_per_period_schema,
          market_wide_schema, validate_* helpers, assert_*_primary_key_unique
@paper_ref: N/A
@data_flow: per-article scores -> per-(ticker, period) aggregates -> market-wide

Canonical pandera schemas for the S2 sentiment layer. Mirrors the conventions
established in :mod:`equity.data.schema` (S1): strict columns, tz-aware
timestamp dtypes, separate ``validate_*`` and ``assert_*_primary_key_unique``
helpers, defensive tz-awareness guards.

Timezone semantics
-------------------
All sentiment timestamps are stored tz-aware UTC, matching the
``articles_joined.parquet`` convention from S1.3 (both ``period_close_ts`` and
``published_at`` canonicalized to UTC). ``period_close_ts`` here is the same
NYSE session close (16:00 / 13:00 ET) used as the join key in S1.3, just
viewed in UTC.

Probabilities invariant
-----------------------
For every row, ``pos + neg + neu`` must sum to 1 (within tolerance). This is
enforced by :func:`validate_sentiment_per_article` /
:func:`validate_sentiment_per_period` /
:func:`validate_market_wide` via :func:`_assert_probs_sum_to_one`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa

# Storage timezone -- matches S1's articles_joined convention (both columns
# UTC). Aggregation compares timestamps in this single tz to avoid DST
# pitfalls.
SENTIMENT_TZ = "UTC"

# Tolerance for the ``pos + neg + neu == 1`` invariant. FinBERT softmax sums
# to 1 up to float32 rounding; VADER's neg/neu/pos also sum to 1; allow a
# generous epsilon to absorb float arithmetic.
_PROB_SUM_TOL = 1e-6

# Canonical column order for the per-article score cache. NOTE: ``ticker``
# and ``published_at`` are intentionally ABSENT -- they are pass-through from
# the input articles frame at read time (see
# :meth:`SentimentCache.score_articles`), so the cache parquet stores ONLY
# the score-result surface (article_key + model metadata + probabilities +
# score). This eliminates ticker drift on warm-path cache hits (FOC-49 B2):
# the same text re-scored under a different ticker returns the new ticker
# from the input, never the stale cached ticker. ``article_key`` already
# hashes (text + model_name + model_revision), so the score-result is
# reproducible from the cache alone.
CANONICAL_PER_ARTICLE_COLUMNS: list[str] = [
    "article_key",
    "model_name",
    "model_revision",
    "pos",
    "neg",
    "neu",
    "score",
]

# Canonical column order for the per-(ticker, period) aggregate.
CANONICAL_PER_PERIOD_COLUMNS: list[str] = [
    "ticker",
    "period_close_ts",
    "sentiment_score",
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_neu",
    "n_articles",
]

# Canonical column order for the market-wide aggregate (no ticker col).
CANONICAL_MARKET_WIDE_COLUMNS: list[str] = [
    "period_close_ts",
    "sentiment_score",
    "sentiment_pos",
    "sentiment_neg",
    "sentiment_neu",
    "n_articles",
]


def _assert_probs_sum_to_one(df: pd.DataFrame, pos: str, neg: str, neu: str) -> None:
    """Assert ``pos + neg + neu`` is approximately 1 for every row.

    Defensive guard -- FinBERT softmax / VADER polarity scores sum to 1 by
    construction; a violation here indicates a corrupted cache or a custom
    scorer returning unnormalized probabilities. No-op on empty frames.

    NaN is treated as a violation: a NaN-producing scorer would otherwise
    bypass the comparison (NaN comparisons are always False) and silently
    propagate NaN downstream, defeating the PIT guard invariants (FOC-49
    round-3 review blocker).
    """
    if df.empty:
        return
    total = df[pos].to_numpy() + df[neg].to_numpy() + df[neu].to_numpy()
    bad_mask = (total < 1.0 - _PROB_SUM_TOL) | (total > 1.0 + _PROB_SUM_TOL) | np.isnan(total)
    if bad_mask.any():
        bad_idx = df.index[bad_mask][:5].tolist()
        sample = df.loc[bad_idx, [pos, neg, neu]].to_dict("records")
        raise ValueError(
            f"probabilities invariant violated: pos+neg+neu != 1 for "
            f"{int(bad_mask.sum())} row(s). First 5: {sample}"
        )


# ---------------------------------------------------------------------------
# Per-article schema (cache)
# ---------------------------------------------------------------------------

sentiment_per_article_schema: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        "article_key": pa.Column(str, coerce=True),
        "model_name": pa.Column(str, coerce=True),
        "model_revision": pa.Column(str, coerce=True),
        # nullable=False: a NaN-producing scorer must be rejected at schema
        # validation, not just by the prob-sum custom check (FOC-49 round-3
        # review blocker -- NaN bypasses the comparison-based guard).
        "pos": pa.Column(float, coerce=True, nullable=False),
        "neg": pa.Column(float, coerce=True, nullable=False),
        "neu": pa.Column(float, coerce=True, nullable=False),
        "score": pa.Column(float, coerce=True, nullable=False),
    },
    strict=True,
    coerce=False,
)


def validate_sentiment_per_article(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a per-article sentiment frame against
    :data:`sentiment_per_article_schema` and the ``pos+neg+neu==1`` invariant.

    This validates the CACHE-INTERNAL canonical frame (score-result surface
    only, no ``ticker`` / ``published_at``). The frame returned to callers by
    :meth:`SentimentCache.score_articles` carries additional pass-through
    input columns and is NOT validated against this strict schema (callers
    consume specific columns by name).
    """
    validated = sentiment_per_article_schema.validate(df)
    _assert_probs_sum_to_one(validated, "pos", "neg", "neu")
    return validated


def assert_per_article_primary_key_unique(df: pd.DataFrame) -> None:
    """Assert ``article_key`` is unique in ``df`` (primary key of the cache)."""
    if df.empty:
        return
    dup_mask = df.duplicated(subset=["article_key"], keep=False)
    if dup_mask.any():
        dups = df.loc[dup_mask, ["article_key"]].head(5)
        raise ValueError(
            f"Primary key (article_key) has {int(dup_mask.sum())} duplicate "
            f"rows. First 5:\n{dups.to_string(index=False)}"
        )


# ---------------------------------------------------------------------------
# Per-(ticker, period) aggregate schema
# ---------------------------------------------------------------------------

sentiment_per_period_schema: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        "ticker": pa.Column(str, coerce=True),
        "period_close_ts": pa.Column(
            pd.DatetimeTZDtype(tz=SENTIMENT_TZ),
            coerce=False,
        ),
        # nullable=False on probabilities + score: NaN must not silently
        # propagate downstream (FOC-49 round-3 review blocker).
        "sentiment_score": pa.Column(float, coerce=True, nullable=False),
        "sentiment_pos": pa.Column(float, coerce=True, nullable=False),
        "sentiment_neg": pa.Column(float, coerce=True, nullable=False),
        "sentiment_neu": pa.Column(float, coerce=True, nullable=False),
        "n_articles": pa.Column(int, coerce=True),
    },
    strict=True,
    coerce=False,
)


def validate_sentiment_per_period(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a per-(ticker, period) aggregate frame and the
    ``pos+neg+neu==1`` invariant.
    """
    validated = sentiment_per_period_schema.validate(df)
    if not validated.empty:
        ts = validated["period_close_ts"]
        if ts.dt.tz is None:
            raise ValueError(
                f"period_close_ts must be timezone-aware ({SENTIMENT_TZ}). "
                "Received tz-naive timestamps; aggregation corruption likely."
            )
    _assert_probs_sum_to_one(validated, "sentiment_pos", "sentiment_neg", "sentiment_neu")
    return validated


def assert_per_period_primary_key_unique(df: pd.DataFrame) -> None:
    """Assert ``(ticker, period_close_ts)`` is unique in ``df``."""
    if df.empty:
        return
    dup_mask = df.duplicated(subset=["ticker", "period_close_ts"], keep=False)
    if dup_mask.any():
        dups = df.loc[dup_mask, ["ticker", "period_close_ts"]].head(5)
        raise ValueError(
            f"Primary key (ticker, period_close_ts) has "
            f"{int(dup_mask.sum())} duplicate rows. First 5:\n"
            f"{dups.to_string(index=False)}"
        )


# ---------------------------------------------------------------------------
# Market-wide aggregate schema (no ticker col)
# ---------------------------------------------------------------------------

market_wide_schema: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        "period_close_ts": pa.Column(
            pd.DatetimeTZDtype(tz=SENTIMENT_TZ),
            coerce=False,
        ),
        # nullable=False on probabilities + score (FOC-49 round-3 review blocker).
        "sentiment_score": pa.Column(float, coerce=True, nullable=False),
        "sentiment_pos": pa.Column(float, coerce=True, nullable=False),
        "sentiment_neg": pa.Column(float, coerce=True, nullable=False),
        "sentiment_neu": pa.Column(float, coerce=True, nullable=False),
        "n_articles": pa.Column(int, coerce=True),
    },
    strict=True,
    coerce=False,
)


def validate_market_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a market-wide aggregate frame and the ``pos+neg+neu==1``
    invariant.
    """
    validated = market_wide_schema.validate(df)
    if not validated.empty:
        ts = validated["period_close_ts"]
        if ts.dt.tz is None:
            raise ValueError(
                f"period_close_ts must be timezone-aware ({SENTIMENT_TZ}). "
                "Received tz-naive timestamps; aggregation corruption likely."
            )
    _assert_probs_sum_to_one(validated, "sentiment_pos", "sentiment_neg", "sentiment_neu")
    return validated


def assert_market_wide_primary_key_unique(df: pd.DataFrame) -> None:
    """Assert ``period_close_ts`` is unique in ``df``."""
    if df.empty:
        return
    dup_mask = df.duplicated(subset=["period_close_ts"], keep=False)
    if dup_mask.any():
        dups = df.loc[dup_mask, ["period_close_ts"]].head(5)
        raise ValueError(
            f"Primary key (period_close_ts) has {int(dup_mask.sum())} "
            f"duplicate rows. First 5:\n{dups.to_string(index=False)}"
        )


__all__ = [
    "SENTIMENT_TZ",
    "CANONICAL_PER_ARTICLE_COLUMNS",
    "CANONICAL_PER_PERIOD_COLUMNS",
    "CANONICAL_MARKET_WIDE_COLUMNS",
    "sentiment_per_article_schema",
    "sentiment_per_period_schema",
    "market_wide_schema",
    "validate_sentiment_per_article",
    "validate_sentiment_per_period",
    "validate_market_wide",
    "assert_per_article_primary_key_unique",
    "assert_per_period_primary_key_unique",
    "assert_market_wide_primary_key_unique",
]
