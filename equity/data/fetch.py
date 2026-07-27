"""
@module: equity.data.fetch
@depends: yfinance, pandas
@exports: fetch_yfinance_ohlcv
@paper_ref: N/A
@data_flow: tickers + window -> yfinance (in-process) -> long-form OHLCV frame

yfinance wrapper producing the canonical long-form OHLCV DataFrame consumed by
:func:`equity.data.loader.EquityDataLoader.fetch_prices`.

Timezone semantics
-------------------
``yfinance.Ticker.history(interval="1d")`` returns a DataFrame indexed by a
tz-aware ``America/New_York`` DatetimeIndex; the index value for a given
session is the session close (16:00 ET for US daily bars). We promote that
index to the ``period_close_ts`` column verbatim -- never convert to UTC.

VWAP proxy
-----------
yfinance does not return VWAP for daily bars. We compute the standard proxy
``vwap = (high + low + close) / 3`` and document it here. If a future source
provides a true VWAP, replace the computation in :func:`_add_vwap`.

Error handling
---------------
Per-ticker errors (delisted within the window, no data, transient network) are
logged and the ticker is skipped -- the batch never crashes on a single
ticker. An empty input ticker list returns an empty canonical frame.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf

from equity.data.schema import CANONICAL_PRICE_COLUMNS

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


def _empty_canonical() -> pd.DataFrame:
    """Return an empty DataFrame with the canonical columns (object dtype).

    Used as the "no data" sentinel. Dtypes are intentionally loose here; the
    real validation happens in :func:`equity.data.schema.validate_prices`,
    which is only invoked on non-empty frames by :meth:`fetch_prices`.
    """
    return pd.DataFrame(columns=CANONICAL_PRICE_COLUMNS)


def _add_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ``vwap = (high + low + close) / 3`` if missing.

    yfinance does not return VWAP for daily bars; this is the standard proxy.
    The result is left as ``float`` (matches the canonical schema) and is
    ``NaN`` where any of high/low/close is ``NaN`` (partial bars).
    """
    if "vwap" not in df.columns or df["vwap"].isna().all():
        df = df.copy()
        df["vwap"] = (df["high"] + df["low"] + df["close"]) / 3.0
    return df


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
        Window bounds (inclusive). Date-like strings or ``datetime.date``.
        Passed to yfinance as ``YYYY-MM-DD`` strings.
    period:
        Bar interval forwarded to yfinance (e.g. ``"1d"`` for daily bars).

    Returns
    -------
    pd.DataFrame
        Long-form frame with the canonical 9 columns
        (see :data:`equity.data.schema.CANONICAL_PRICE_COLUMNS`).
        ``period_close_ts`` is tz-aware ``America/New_York`` (session close).
        Returns an empty canonical frame if no tickers or no data.

    Notes
    -----
    Per-ticker errors are logged and skipped; the batch never raises on a
    single ticker. yfinance is called in-process via ``Ticker.history`` (no
    subprocess).
    """
    if not tickers:
        return _empty_canonical()

    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end).strftime("%Y-%m-%d")

    frames: list[pd.DataFrame] = []
    for tk in tickers:
        try:
            hist = yf.Ticker(tk).history(
                start=start_str,
                end=end_str,
                interval=period,
                auto_adjust=False,  # keep the original Adj Close column
                actions=False,  # no dividends/splits
                raise_errors=True,
            )
        except Exception as exc:  # noqa: BLE001 - yfinance raises various errors
            log.warning(
                "yfinance: %s failed in [%s, %s]: %s", tk, start_str, end_str, exc
            )
            continue
        if hist is None or hist.empty:
            log.info(
                "yfinance: no data for %s in [%s, %s]", tk, start_str, end_str
            )
            continue

        # hist.index is a tz-aware America/New_York DatetimeIndex (session
        # close for daily bars). Promote it to the period_close_ts column.
        out = pd.DataFrame(
            {
                "ticker": tk,
                "period_close_ts": hist.index,
                "open": hist.get("Open"),
                "high": hist.get("High"),
                "low": hist.get("Low"),
                "close": hist.get("Close"),
                "adj_close": hist.get("Adj Close"),
                "volume": hist.get("Volume"),
            }
        ).reset_index(drop=True)
        out = _add_vwap(out)
        frames.append(out)

    if not frames:
        return _empty_canonical()

    long_df = pd.concat(frames, ignore_index=True)
    return long_df[CANONICAL_PRICE_COLUMNS]


# ---------------------------------------------------------------------------
# S1.3: articles ingestion (seed-based; NO network)
# ---------------------------------------------------------------------------

from pathlib import Path

from equity.data.schema import ARTICLE_TZ, CANONICAL_ARTICLE_COLUMNS


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
