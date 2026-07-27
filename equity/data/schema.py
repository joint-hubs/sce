"""
@module: equity.data.schema
@depends: pandera, pandas
@exports: CANONICAL_PRICE_COLUMNS, prices_schema, validate_prices, assert_primary_key_unique
@paper_ref: N/A
@data_flow: raw OHLCV DataFrame -> pandera schema validation -> tz-aware prices frame

Canonical schema for ``prices.parquet``. This is the FIRST use of pandera in
the repo and the FIRST tz-aware timestamp code -- the conventions established
here are reused by S1.3 (point-in-time text join) and later slices.

Timezone semantics
-------------------
``period_close_ts`` is the exchange-local session close (16:00 ET for US daily
bars). yfinance returns a tz-aware ``America/New_York`` DatetimeIndex; we
preserve it as-is and REQUIRE tz-awareness at validation time. Tz-naive values
are rejected -- this is the guard that prevents accidental UTC writes that
would break the point-in-time join in S1.3.

The tz-awareness guard is enforced by declaring the column dtype as
``pd.DatetimeTZDtype(tz="America/New_York")`` with ``coerce=False``: a tz-naive
``datetime64[ns]`` Series fails the dtype check. ``validate_prices`` adds a
redundant ``dt.tz is not None`` assertion for documentation/defense (the dtype
check is the primary guard).
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

# Canonical column order for prices.parquet. The 9 columns below are the data
# columns; ``year``/``month`` Hive partition keys are added on read-back (see
# ``EquityDataLoader.fetch_prices``).
CANONICAL_PRICE_COLUMNS: list[str] = [
    "ticker",
    "period_close_ts",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "hlc_average",
]

# Exchange-local timezone for US daily bars. yfinance returns America/New_York
# timestamps; we store them verbatim (never convert to UTC).
EXCHANGE_TZ = "America/New_York"

# Pandera schema for the canonical prices frame.
#
# OHLCV fields are nullable to accommodate partial bars for delisted tickers
# within the window (yfinance may return NaNs near the delisting event). The
# ``ticker`` and ``period_close_ts`` primary key is non-nullable and enforced
# separately by :func:`assert_primary_key_unique`.
#
# Note: schema-level ``coerce`` is False so that the tz-aware datetime dtype
# check on ``period_close_ts`` is not stripped; float columns coerce
# individually.
prices_schema: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        "ticker": pa.Column(str, coerce=True),
        "period_close_ts": pa.Column(
            pd.DatetimeTZDtype(tz=EXCHANGE_TZ),
            coerce=False,
        ),
        "open": pa.Column(float, nullable=True, coerce=True),
        "high": pa.Column(float, nullable=True, coerce=True),
        "low": pa.Column(float, nullable=True, coerce=True),
        "close": pa.Column(float, nullable=True, coerce=True),
        "adj_close": pa.Column(float, nullable=True, coerce=True),
        "volume": pa.Column(float, nullable=True, coerce=True),
        "hlc_average": pa.Column(float, nullable=True, coerce=True),
    },
    strict=True,
    coerce=False,
)


def validate_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a prices DataFrame against :data:`prices_schema` and the
    tz-awareness rule on ``period_close_ts``.

    Parameters
    ----------
    df:
        Long-form DataFrame with the canonical 9 columns (see
        :data:`CANONICAL_PRICE_COLUMNS`).

    Returns
    -------
    pd.DataFrame
        The validated, coerced DataFrame.

    Raises
    ------
    pandera.errors.SchemaError
        On schema violations (missing/extra columns, wrong dtype, etc.).
        Tz-naive ``period_close_ts`` fails here via the
        ``DatetimeTZDtype`` dtype check.
    ValueError
        If ``period_close_ts`` is empty-but-tz-naive (defensive guard; the
        dtype check is the primary guard for non-empty frames).
    """
    validated = prices_schema.validate(df)
    # Defensive tz-awareness guard. The dtype check above already rejects
    # tz-naive non-empty frames; this handles edge cases (e.g. an empty frame
    # that somehow slipped through with object dtype) and documents intent.
    if not validated.empty:
        ts = validated["period_close_ts"]
        if ts.dt.tz is None:
            raise ValueError(
                f"period_close_ts must be timezone-aware ({EXCHANGE_TZ}). "
                "Received tz-naive timestamps; yfinance should return "
                "America/New_York-aware values -- check the fetch wrapper."
            )
    return validated


def assert_primary_key_unique(df: pd.DataFrame) -> None:
    """Assert ``(ticker, period_close_ts)`` is unique in ``df``.

    This is the primary key of ``prices.parquet``. Raises :class:`ValueError`
    with a small sample of duplicates on failure. No-op on empty frames.
    """
    if df.empty:
        return
    dup_mask = df.duplicated(subset=["ticker", "period_close_ts"], keep=False)
    if dup_mask.any():
        dups = df.loc[dup_mask, ["ticker", "period_close_ts"]].head(5)
        raise ValueError(
            f"Primary key (ticker, period_close_ts) has {int(dup_mask.sum())} "
            f"duplicate rows. First 5:\n{dups.to_string(index=False)}"
        )


# ---------------------------------------------------------------------------
# S1.3: articles schema (point-in-time text layer)
# ---------------------------------------------------------------------------

# Canonical column order for articles.parquet. The 4 data columns below are the
# data columns; ``year``/``month`` Hive partition keys are added on write (see
# ``EquityDataLoader.fetch_articles``).
CANONICAL_ARTICLE_COLUMNS: list[str] = [
    "ticker",
    "published_at",
    "text",
    "source",
]

# Canonical storage timezone for ``published_at``. RSS/Kaggle feeds publish in
# UTC; ``pd.Timestamp(...).tz_convert("UTC")`` is idempotent for already-UTC
# timestamps. We store the column as ``DatetimeTZDtype(tz="UTC")`` everywhere
# and compare with the price-side ``period_close_ts`` (America/New_York) in
# UTC -- see ``join_articles_to_prices``.
ARTICLE_TZ = "UTC"

# Pandera schema for the canonical articles frame.
#
# ``ticker`` and ``source`` are non-null strings; ``published_at`` is a
# tz-aware UTC datetime (the primary guard against point-in-time leakage);
# ``text`` is nullable (some sources ship a headline-only payload). The
# ``ticker``/``published_at``/``source`` triple is the primary key and is
# enforced separately by :func:`assert_articles_primary_key_unique`.
articles_schema: pa.DataFrameSchema = pa.DataFrameSchema(
    {
        "ticker": pa.Column(str, coerce=True),
        "published_at": pa.Column(
            pd.DatetimeTZDtype(tz=ARTICLE_TZ),
            coerce=False,
        ),
        "text": pa.Column(str, nullable=True, coerce=True),
        "source": pa.Column(str, coerce=True),
    },
    strict=True,
    coerce=False,
)


def validate_articles(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an articles DataFrame against :data:`articles_schema` and the
    tz-awareness rule on ``published_at``.

    Parameters
    ----------
    df:
        Long-form DataFrame with the canonical 4 columns (see
        :data:`CANONICAL_ARTICLE_COLUMNS`).

    Returns
    -------
    pd.DataFrame
        The validated, coerced DataFrame.

    Raises
    ------
    pandera.errors.SchemaError
        On schema violations (missing/extra columns, wrong dtype, etc.).
        Tz-naive ``published_at`` fails here via the ``DatetimeTZDtype``
        dtype check.
    ValueError
        If ``published_at`` is empty-but-tz-naive (defensive guard; the dtype
        check is the primary guard for non-empty frames).
    """
    validated = articles_schema.validate(df)
    # Defensive tz-awareness guard. The dtype check above already rejects
    # tz-naive non-empty frames; this handles edge cases and documents intent.
    if not validated.empty:
        ts = validated["published_at"]
        if ts.dt.tz is None:
            raise ValueError(
                f"published_at must be timezone-aware ({ARTICLE_TZ}). "
                "Received tz-naive timestamps; RSS/Kaggle feeds publish in "
                "UTC -- check the seed loader / fetch wrapper."
            )
    return validated


def assert_articles_primary_key_unique(df: pd.DataFrame) -> None:
    """Assert ``(ticker, published_at, source)`` is unique in ``df``.

    This is the primary key of ``articles.parquet`` (dedup same article from
    same source). Raises :class:`ValueError` with a small sample of
    duplicates on failure. No-op on empty frames.
    """
    if df.empty:
        return
    dup_mask = df.duplicated(subset=["ticker", "published_at", "source"], keep=False)
    if dup_mask.any():
        dups = df.loc[
            dup_mask, ["ticker", "published_at", "source"]
        ].head(5)
        raise ValueError(
            f"Primary key (ticker, published_at, source) has "
            f"{int(dup_mask.sum())} duplicate rows. First 5:\n"
            f"{dups.to_string(index=False)}"
        )


__all__ = [
    "CANONICAL_PRICE_COLUMNS",
    "EXCHANGE_TZ",
    "prices_schema",
    "validate_prices",
    "assert_primary_key_unique",
    "CANONICAL_ARTICLE_COLUMNS",
    "ARTICLE_TZ",
    "articles_schema",
    "validate_articles",
    "assert_articles_primary_key_unique",
]
