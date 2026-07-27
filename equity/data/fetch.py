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


__all__ = ["fetch_yfinance_ohlcv"]
