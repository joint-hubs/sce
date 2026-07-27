"""
@module: equity.data.fetch
@depends: yfinance, pandas, pandas_market_calendars
@exports: fetch_yfinance_ohlcv, fetch_articles_from_seed
@paper_ref: N/A
@data_flow: tickers + window -> yfinance (in-process) -> long-form OHLCV frame

yfinance wrapper producing the canonical long-form OHLCV DataFrame consumed by
:func:`equity.data.loader.EquityDataLoader.fetch_prices`.

Timezone semantics
-------------------
``yfinance.Ticker.history(interval="1d")`` returns a DataFrame indexed by a
tz-aware ``America/New_York`` DatetimeIndex, but the index value for a given
session is the **exchange-local midnight (00:00 ET)**, NOT the session close.
To make ``prices.parquet.period_close_ts`` a valid key consumable by the
point-in-time join in S1.3 (which derives ``period_close_ts`` from the XNYS
``market_close`` -- 16:00 ET, or 13:00 ET on early-close sessions), we
**canonicalize** the index to the XNYS session close for each date. The
00:00 ET index value is replaced by the 16:00 / 13:00 ET close from
``pandas_market_calendars``. See :func:`_canonicalize_session_close`.

HLC average (NOT VWAP)
-----------------------
yfinance does not return VWAP for daily bars. We compute the standard
``(high + low + close) / 3`` proxy but name the column ``hlc_average`` (NOT
``vwap``) -- the formula is a simple average, not a volume-weighted value.
Renaming prevents S2 feature engineering from misinterpreting the column as
true VWAP. See :func:`_add_hlc_average`.

Error handling
---------------
Per-ticker errors (delisted within the window, no data, transient network) are
logged and the ticker is skipped -- the batch never crashes on a single
ticker. An empty input ticker list returns an empty canonical frame. A
``None`` frame from yfinance (possible fetch error masked by yfinance
internals) is logged as a WARNING, not silently treated as "no data".
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from equity.data.schema import (
    ARTICLE_TZ,
    CANONICAL_ARTICLE_COLUMNS,
    CANONICAL_PRICE_COLUMNS,
    EXCHANGE_TZ,
)

log = logging.getLogger(__name__)

# Map yfinance's PascalCase field names to the canonical lowercase columns.
# ``Adj Close`` is only returned when ``auto_adjust=False``.
_YF_FIELD_MAP: dict[str, str] = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}

# Sentinel marker file written into each parquet store directory by the loader.
# Used by the destructive-rmtree containment guard (see loader._safe_rmtree).
STORE_MARKER = ".equity_store"


def _empty_canonical() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical columns (object dtype).

    Used as the "no data" sentinel. Dtypes are intentionally loose here; the
    real validation happens in :func:`equity.data.schema.validate_prices`,
    which is only invoked on non-empty frames by :meth:`fetch_prices`.
    """
    return pd.DataFrame(columns=CANONICAL_PRICE_COLUMNS)


def _add_hlc_average(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ``hlc_average = (high + low + close) / 3`` if the column is
    absent.

    yfinance does not return VWAP for daily bars; this is the standard HLC
    average proxy (NOT a volume-weighted value -- hence the honest column
    name). The result is left as ``float`` (matches the canonical schema) and
    is ``NaN`` where any of high/low/close is ``NaN`` (partial bars).
    """
    if "hlc_average" not in df.columns:
        df = df.copy()
        df["hlc_average"] = (df["high"] + df["low"] + df["close"]) / 3.0
    return df


def _canonicalize_session_close(
    index: pd.DatetimeIndex,
    schedule: pd.DataFrame,
    period: str = "1d",
) -> tuple[pd.DatetimeIndex, pd.Series]:
    """Map each exchange-local-midnight (00:00 ET) index value to the XNYS
    session-close timestamp for its date.

    yfinance daily bars are indexed at 00:00 ET (exchange-local midnight), not
    the 16:00 ET session close. The point-in-time join in
    :meth:`EquityDataLoader.join_articles_to_prices` derives ``period_close_ts``
    from the XNYS ``market_close`` (16:00 ET, or 13:00 ET on early-close
    sessions), so a 00:00-keyed ``prices.parquet`` would never align as a
    foreign key. This function canonicalizes the fetch index to the XNYS
    close for each session date, making ``prices.parquet.period_close_ts`` a
    valid key the join can hit.

    Parameters
    ----------
    index:
        tz-aware ``America/New_York`` DatetimeIndex from yfinance (00:00 ET
        values).
    schedule:
        Pre-built XNYS schedule DataFrame (``pandas_market_calendars``
        ``.schedule(...)`` output) covering the full fetch window. The
        schedule is computed ONCE per ``fetch_yfinance_ohlcv`` call and shared
        across all tickers (review round 2, suggestion #5: previously
        ``cal.schedule()`` was invoked per-ticker with the same date range).
        Must contain a ``market_close`` column.
    period:
        Bar interval (only ``"1d"`` is canonicalized; intraday periods are
        returned unchanged).

    Returns
    -------
    (canonicalized_index, keep_mask)
        ``canonicalized_index`` is a tz-aware ``America/New_York``
        DatetimeIndex aligned 1:1 with the input ``index`` (``NaT`` where the
        input date has no XNYS session -- e.g. a holiday that slipped through
        yfinance; the caller drops those rows via ``keep_mask``).
        ``keep_mask`` is a boolean Series (aligned to ``index``) that is
        ``True`` for rows whose date has an XNYS session.
    """
    if period != "1d" or len(index) == 0:
        return index, pd.Series([True] * len(index), index=index)
    # Normalize each index value to its calendar date (00:00 ET -> date).
    # ``index.normalize()`` is the DatetimeIndex vectorized method (review
    # round 2, nitpick #11: replaces the O(N) ``[pd.Timestamp(ts).normalize()
    # for ts in index]`` list comprehension).
    dates = index.normalize()
    # market_close is tz-aware America/New_York; build a date -> close map.
    closes_by_date = {}
    for ts in schedule["market_close"]:
        closes_by_date[pd.Timestamp(ts).normalize()] = ts
    mapped = [closes_by_date.get(d) for d in dates]
    keep = [ts is not None for ts in mapped]
    cleaned = [ts if ts is not None else pd.NaT for ts in mapped]
    out_idx = pd.DatetimeIndex(cleaned, tz=EXCHANGE_TZ)
    return out_idx, pd.Series(keep, index=index)


def fetch_yfinance_ohlcv(
    tickers: list[str],
    start: str | date | pd.Timestamp,
    end: str | date | pd.Timestamp,
    period: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV bars from yfinance for the given tickers and window.

    Parameters
    ----------
    tickers:
        List of ticker symbols (e.g. ``["AAPL", "MSFT"]``). Tickers may
        contain dots (e.g. ``"BRK.B"``).
    start, end:
        Window bounds (**inclusive**). Date-like strings or ``datetime.date``.
        Passed to yfinance as ``YYYY-MM-DD`` strings; the ``end`` is offset
        by +1 day before passing because yfinance treats ``end`` as
        **exclusive**.
    period:
        Bar interval forwarded to yfinance (e.g. ``"1d"`` for daily bars).

    Returns
    -------
    pd.DataFrame
        Long-form frame with the canonical 9 columns
        (see :data:`equity.data.schema.CANONICAL_PRICE_COLUMNS`).
        ``period_close_ts`` is the **XNYS session close** (16:00 / 13:00 ET),
        tz-aware ``America/New_York`` (canonicalized from yfinance's 00:00 ET
        index). Returns an empty canonical frame if no tickers or no data.

    Notes
    -----
    Per-ticker errors are logged and skipped; the batch never raises on a
    single ticker. yfinance is called in-process via ``Ticker.history`` (no
    subprocess).
    """
    if not tickers:
        return _empty_canonical()

    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
    # yfinance treats `end` as EXCLUSIVE; the loader's `end` is inclusive, so
    # offset by +1 day to include the last requested day.
    end_str = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Lazily build the XNYS calendar once per call (used to canonicalize the
    # 00:00 ET yfinance index to the 16:00/13:00 ET session close). The
    # schedule is computed ONCE for the full fetch window (review round 2,
    # suggestion #5: previously ``cal.schedule()`` was invoked per-ticker with
    # the same date range -- all tickers share ``[start, end+1d]``, so the
    # schedule is identical).
    schedule: pd.DataFrame | None = None
    if period == "1d":
        from pandas_market_calendars import get_calendar

        cal = get_calendar("XNYS")
        schedule = cal.schedule(start_str, end_str, tz=EXCHANGE_TZ)

    # yfinance raises ``YFException`` subclasses (YFTickerMissingError,
    # YFPricesMissingError, YFRateLimitError, ...) when a fetch fails; the
    # pinned version (>=0.2.40,<2) honors these for interval="1d". We catch
    # broadly and log the exception type explicitly so a non-yfinance error is
    # visible in the log (rather than silently swallowed as "no data").
    frames: list[pd.DataFrame] = []
    for tk in tickers:
        try:
            hist = yf.Ticker(tk).history(
                start=start_str,
                end=end_str,
                interval=period,
                auto_adjust=False,  # keep the original Adj Close column
                actions=False,  # no dividends/splits
                raise_errors=True,  # pinned version honors this for "1d"
            )
        except Exception as exc:  # noqa: BLE001 - yfinance raises various errors
            log.warning(
                "yfinance: %s failed in [%s, %s] (%s): %s",
                tk,
                start_str,
                end_str,
                type(exc).__name__,
                exc,
            )
            continue
        if hist is None:
            # Possible fetch error masked by yfinance internals -- log at
            # WARNING (not silently as "no data") so the operator notices.
            log.warning(
                "yfinance: returned None for %s in [%s, %s] "
                "(possible masked fetch error; skipping ticker).",
                tk,
                start_str,
                end_str,
            )
            continue
        if hist.empty:
            log.info(
                "yfinance: no data for %s in [%s, %s]", tk, start_str, end_str
            )
            continue

        idx = hist.index
        if schedule is not None:
            idx, keep = _canonicalize_session_close(idx, schedule, period=period)
            # Drop rows whose date has no XNYS session (holiday that slipped
            # through yfinance) from BOTH the index and the OHLCV columns so
            # the lengths stay aligned.
            keep_arr = keep.to_numpy()
            hist = hist.loc[keep_arr]
            idx = idx[keep_arr]
        # idx is a tz-aware America/New_York DatetimeIndex; for daily bars it
        # is canonicalized to the session close (16:00/13:00 ET).
        out = pd.DataFrame(
            {
                "ticker": tk,
                "period_close_ts": idx,
                "open": hist.get("Open").to_numpy(),
                "high": hist.get("High").to_numpy(),
                "low": hist.get("Low").to_numpy(),
                "close": hist.get("Close").to_numpy(),
                "adj_close": hist.get("Adj Close").to_numpy(),
                "volume": hist.get("Volume").to_numpy(),
            }
        ).reset_index(drop=True)
        out = _add_hlc_average(out)
        frames.append(out)

    if not frames:
        return _empty_canonical()

    long_df = pd.concat(frames, ignore_index=True)
    return long_df[CANONICAL_PRICE_COLUMNS]


# ---------------------------------------------------------------------------
# S1.3: articles ingestion (seed-based; NO network)
# ---------------------------------------------------------------------------


def _empty_articles_canonical() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical articles columns."""
    return pd.DataFrame(columns=CANONICAL_ARTICLE_COLUMNS)


def fetch_articles_from_seed(seed_path: str | Path) -> pd.DataFrame:
    """Load articles from a committed seed CSV and coerce ``published_at`` to
    tz-aware UTC.

    The seed CSV has the header ``ticker,published_at,text,source`` and an
    optional ``#`` comment line documenting its semantics. ``published_at`` is
    parsed as ISO-8601 (with or without a trailing ``Z`` / offset); values that
    carry an offset are converted to UTC, values that are tz-naive are localized
    to UTC. The output frame has the 4 canonical columns in order (see
    :data:`equity.data.schema.CANONICAL_ARTICLE_COLUMNS`).

    This function performs **no schema validation** -- the caller (typically
    :meth:`EquityDataLoader.fetch_articles`) runs
    :func:`equity.data.schema.validate_articles` before writing the parquet
    store.

    Parameters
    ----------
    seed_path:
        Path to the seed CSV (e.g. ``configs/equity/articles_seed.csv``).

    Returns
    -------
    pd.DataFrame
        Long-form frame with the canonical 4 columns;
        ``published_at`` is tz-aware ``UTC``. Returns an empty canonical
        frame if the seed contains only a header / comment lines.

    Raises
    ------
    FileNotFoundError
        If ``seed_path`` does not exist.
    ValueError
        If the seed is missing any of the 4 required columns.
    """
    path = Path(seed_path)
    if not path.exists():
        raise FileNotFoundError(f"Articles seed file not found: {path}")
    raw = pd.read_csv(path, comment="#")
    if raw.empty:
        return _empty_articles_canonical()
    for col in ("ticker", "published_at", "text", "source"):
        if col not in raw.columns:
            raise ValueError(
                f"Articles seed '{path.name}' missing required column '{col}'."
            )
    out = pd.DataFrame(
        {
            "ticker": raw["ticker"].astype(str),
            # pd.to_datetime with utc=True parses mixed-aware/naive ISO strings
            # and returns a tz-aware UTC Series. Strings already carrying an
            # offset (e.g. "...+00:00") are converted; tz-naive strings are
            # localized to UTC.
            "published_at": pd.to_datetime(raw["published_at"], utc=True),
            "text": raw["text"].astype(object),
            "source": raw["source"].astype(str),
        }
    )
    # Force the exact canonical dtype (pd.to_datetime(utc=True) yields
    # datetime64[ns, UTC]; the explicit astype is a no-op / idempotent guard).
    out["published_at"] = out["published_at"].astype(pd.DatetimeTZDtype(tz=ARTICLE_TZ))
    return out[CANONICAL_ARTICLE_COLUMNS]


__all__ = ["fetch_yfinance_ohlcv", "fetch_articles_from_seed"]
