"""
@module: equity.data.loader
@depends: pandas, tomllib, equity.data.registry
@exports: EquityDataLoader
@paper_ref: N/A
@data_flow: universe config -> universe_file CSV -> alive-ticker tuples
"""

from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from equity.data.registry import PROJECT_ROOT, get_universe_info

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


def _to_timestamp(value: str | date | pd.Timestamp) -> pd.Timestamp:
    """Coerce a date-like value to a normalized (midnight) pandas Timestamp."""
    return pd.Timestamp(value).normalize()


def _to_pydate(value: Any) -> date | None:
    """Convert a pandas datetime scalar to a :class:`datetime.date`; ``None`` for NaT."""
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).date()


def _is_alive(
    listed_at: Any, delisted_at: Any, start: pd.Timestamp, end: pd.Timestamp
) -> bool:
    """Return True if the listing window overlaps ``[start, end]``.

    Missing ``listed_at`` is treated as negative infinity (always listed in
    the past); missing ``delisted_at`` is treated as positive infinity (still
    listed).
    """
    listed_ok = pd.isna(listed_at) or pd.Timestamp(listed_at) <= end
    delisted_ok = pd.isna(delisted_at) or pd.Timestamp(delisted_at) >= start
    return bool(listed_ok and delisted_ok)


def _dead_reason(
    listed_at: Any, delisted_at: Any, start: pd.Timestamp, end: pd.Timestamp
) -> str:
    """Human-readable reason a ticker is not alive for the window."""
    if not pd.isna(delisted_at) and pd.Timestamp(delisted_at) < start:
        return f"delisted on {pd.Timestamp(delisted_at).date()} before start"
    if not pd.isna(listed_at) and pd.Timestamp(listed_at) > end:
        return f"listed on {pd.Timestamp(listed_at).date()} after end"
    return "outside window"


class EquityDataLoader:
    """Delisting-aware equity universe loader.

    Resolves a universe config (TOML) via :mod:`equity.data.registry`, reads the
    referenced ``universe_file`` CSV, and exposes :meth:`universe` returning the
    set of tickers alive for any part of the ``[start, end]`` window.

    A ticker is "alive for the window" when its listing window overlaps
    ``[start, end]``::

        listed_at <= end AND (delisted_at IS NULL OR delisted_at >= start)

    Missing ``listed_at`` is treated as negative infinity (always listed in the
    past); missing ``delisted_at`` is treated as positive infinity (still
    listed). A ticker delisted DURING the window is INCLUDED with its
    ``delisted_at`` date (survivorship-aware -- never dropped).

    When an explicit ``tickers`` filter is provided, every requested ticker must
    be present in the universe file AND alive for the window; otherwise a
    :class:`ValueError` is raised at construction time naming the offending
    ticker and the reason. When ``tickers`` is ``None`` the full alive set is
    returned with no validation error.

    Parameters
    ----------
    universe:
        Universe name (e.g. ``"sp500"``) resolving to ``configs/equity/<name>.toml``.
    start, end:
        Window bounds (inclusive). Date-like strings or :class:`datetime.date`.
    period:
        Bar interval forwarded to the OHLCV fetcher (e.g. ``"1d"`` for daily
        bars). Used by :meth:`fetch_prices` (S1.2).
    tickers:
        Optional explicit ticker filter. When provided, each ticker is validated
        against the universe file and the window.
    """

    def __init__(
        self,
        universe: str,
        start: str | date,
        end: str | date,
        period: str = "1d",
        tickers: list[str] | None = None,
    ) -> None:
        self.universe_name = universe
        self.start = _to_timestamp(start)
        self.end = _to_timestamp(end)
        self.period = period
        self.tickers = list(tickers) if tickers is not None else None

        self.info = get_universe_info(universe)
        universe_path = self.info.universe_file
        if not universe_path.is_absolute():
            universe_path = PROJECT_ROOT / universe_path
        if not universe_path.exists():
            raise FileNotFoundError(f"Universe file not found: {universe_path}")
        self._universe_path = universe_path

        self._frame = pd.read_csv(universe_path, comment="#")
        for col in ("ticker", "listed_at", "delisted_at"):
            if col not in self._frame.columns:
                raise ValueError(
                    f"Universe file '{universe_path.name}' missing required column '{col}'."
                )
        self._frame["ticker"] = self._frame["ticker"].astype(str)
        for col in ("listed_at", "delisted_at"):
            self._frame[col] = pd.to_datetime(self._frame[col], errors="coerce")
        self._frame = self._frame.drop_duplicates(subset="ticker", keep="first").reset_index(
            drop=True
        )

        if self.tickers is not None:
            self._validate_requested_tickers()

    def _validate_requested_tickers(self) -> None:
        assert self.tickers is not None
        file_tickers = set(self._frame["ticker"])
        indexed = self._frame.set_index("ticker")
        for ticker in self.tickers:
            if ticker not in file_tickers:
                raise ValueError(
                    f"Ticker '{ticker}' is not present in universe file "
                    f"'{self._universe_path.name}'."
                )
            listed_at = indexed.loc[ticker, "listed_at"]
            delisted_at = indexed.loc[ticker, "delisted_at"]
            if not _is_alive(listed_at, delisted_at, self.start, self.end):
                reason = _dead_reason(listed_at, delisted_at, self.start, self.end)
                raise ValueError(
                    f"Ticker '{ticker}' is not alive for window "
                    f"[{self.start.date()}, {self.end.date()}]: {reason}."
                )

    def universe(self) -> list[tuple[str, date | None, date | None]]:
        """Return ``(ticker, listed_at, delisted_at)`` tuples for tickers alive
        in ``[start, end]``.

        When ``tickers`` was provided at construction, the result is restricted
        to that filter (all of which were validated as alive). Otherwise the
        full alive set is returned. A ticker delisted during the window is
        included with its ``delisted_at`` date.
        """
        if self.tickers is not None:
            frame = self._frame[self._frame["ticker"].isin(self.tickers)]
        else:
            frame = self._frame

        listed = frame["listed_at"]
        delisted = frame["delisted_at"]
        alive_mask = (listed.isna() | (listed <= self.end)) & (
            delisted.isna() | (delisted >= self.start)
        )
        alive_frame = frame[alive_mask]

        return [
            (str(ticker), _to_pydate(listed_at), _to_pydate(delisted_at))
            for ticker, listed_at, delisted_at in zip(
                alive_frame["ticker"].tolist(),
                alive_frame["listed_at"].tolist(),
                alive_frame["delisted_at"].tolist(),
            )
        ]

    # ------------------------------------------------------------------
    # S1.2: OHLCV ingestion + partitioned parquet store
    # ------------------------------------------------------------------

    def _load_prices_config(self) -> dict[str, Any]:
        """Load the ``[prices]`` section from the universe TOML config.

        Returns an empty dict if the section is absent (callers apply their
        own defaults). Reads the same TOML that produced :attr:`self.info`.
        """
        with self.info.path.open("rb") as handle:
            payload = tomllib.load(handle)
        return payload.get("prices", {})

    def fetch_prices(self, output_dir: str | Path | None = None) -> Path:
        """Fetch OHLCV bars for the alive universe and write a partitioned
        parquet store.

        Flow:

        1. Resolve the alive-ticker set via :meth:`universe` (delisting-aware).
        2. Fetch OHLCV via :func:`equity.data.fetch.fetch_yfinance_ohlcv`
           (in-process yfinance; per-ticker errors are logged and skipped).
        3. Validate the long-form frame with
           :func:`equity.data.schema.validate_prices` (pandera schema + tz-aware
           ``period_close_ts`` guard) and assert primary-key uniqueness via
           :func:`equity.data.schema.assert_primary_key_unique`.
        4. Write a Hive-style partitioned parquet dataset under ``output_dir``,
           partitioned by ``year``/``month`` derived from ``period_close_ts``
           (America/New_York session close).

        The write is a **full rewrite** (not incremental): if ``output_dir``
        exists it is removed first so stale partitions from prior runs do not
        leak in. On read-back, ``year``/``month`` are reconstructed from the
        Hive path (they are partition keys, not data columns -- see
        ``equity/data/README.md``).

        Parameters
        ----------
        output_dir:
            Output directory for the partitioned parquet store. When ``None``,
            falls back to ``[prices].output_dir`` from the universe TOML
            (default ``data/equity/prices`` relative to ``PROJECT_ROOT``).

        Returns
        -------
        pathlib.Path
            The output directory Path (absolute).

        Raises
        ------
        ValueError
            If the alive universe is empty for the window.
        RuntimeError
            If yfinance returned no rows for any ticker.
        """
        from equity.data.fetch import fetch_yfinance_ohlcv
        from equity.data.schema import (
            assert_primary_key_unique,
            validate_prices,
        )

        prices_cfg = self._load_prices_config()
        if output_dir is not None:
            out_path = Path(output_dir)
        else:
            out_path = Path(prices_cfg.get("output_dir", "data/equity/prices"))
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path

        alive = self.universe()
        tickers = [t for (t, _listed, _delisted) in alive]
        if not tickers:
            raise ValueError(
                f"No alive tickers in universe '{self.universe_name}' for "
                f"window [{self.start.date()}, {self.end.date()}]."
            )

        raw = fetch_yfinance_ohlcv(tickers, self.start, self.end, self.period)
        if raw.empty:
            raise RuntimeError(
                f"yfinance returned no rows for {len(tickers)} tickers in "
                f"[{self.start.date()}, {self.end.date()}]."
            )
        validated = validate_prices(raw)
        assert_primary_key_unique(validated)

        # Derive year/month partition keys from the tz-aware session close.
        out = validated.copy()
        out["year"] = out["period_close_ts"].dt.year
        out["month"] = out["period_close_ts"].dt.month

        # Full rewrite: clear stale partitions before writing.
        if out_path.exists():
            shutil.rmtree(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

        partition_cols = prices_cfg.get("partition_cols", ["year", "month"])
        out.to_parquet(
            out_path,
            partition_cols=partition_cols,
            index=False,
        )
        return out_path


__all__ = ["EquityDataLoader"]
