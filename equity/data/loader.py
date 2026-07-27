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

    # ------------------------------------------------------------------
    # S1.3: articles ingestion + point-in-time join
    # ------------------------------------------------------------------

    def _load_articles_config(self) -> dict[str, Any]:
        """Load the ``[articles]`` section from the universe TOML config.

        Returns an empty dict if the section is absent (callers apply their
        own defaults). Reads the same TOML that produced :attr:`self.info`.
        """
        with self.info.path.open("rb") as handle:
            payload = tomllib.load(handle)
        return payload.get("articles", {})

    def fetch_articles(self, output_dir: str | Path | None = None) -> Path:
        """Load articles from the configured seed CSV and write a partitioned
        parquet store.

        Flow:

        1. Resolve the ``[articles].seed_file`` path from the universe TOML
           (default ``configs/equity/articles_seed.csv``).
        2. Load the seed via :func:`equity.data.fetch.fetch_articles_from_seed`
           (NO network -- live RSS/Kaggle ingestion is S7).
        3. Validate the long-form frame with
           :func:`equity.data.schema.validate_articles` and assert primary-key
           uniqueness via
           :func:`equity.data.schema.assert_articles_primary_key_unique`.
        4. Write a Hive-style partitioned parquet dataset under ``output_dir``,
           partitioned by ``year``/``month`` derived from ``published_at`` (UTC).

        The write is a **full rewrite** (not incremental): if ``output_dir``
        exists it is removed first so stale partitions do not leak in. On
        read-back, ``year``/``month`` are reconstructed from the Hive path.

        Parameters
        ----------
        output_dir:
            Output directory for the partitioned parquet store. When ``None``,
            falls back to ``[articles].output_dir`` from the universe TOML
            (default ``data/equity/articles`` relative to ``PROJECT_ROOT``).

        Returns
        -------
        pathlib.Path
            The output directory Path (absolute).

        Raises
        ------
        ValueError
            If the seed frame is empty after validation.
        """
        from equity.data.fetch import fetch_articles_from_seed
        from equity.data.schema import (
            assert_articles_primary_key_unique,
            validate_articles,
        )

        articles_cfg = self._load_articles_config()
        if output_dir is not None:
            out_path = Path(output_dir)
        else:
            out_path = Path(articles_cfg.get("output_dir", "data/equity/articles"))
        if not out_path.is_absolute():
            out_path = PROJECT_ROOT / out_path

        seed_rel = articles_cfg.get("seed_file", "configs/equity/articles_seed.csv")
        seed_path = Path(seed_rel)
        if not seed_path.is_absolute():
            seed_path = PROJECT_ROOT / seed_path
        raw = fetch_articles_from_seed(seed_path)
        if raw.empty:
            raise ValueError(
                f"Articles seed '{seed_path}' produced an empty frame -- "
                "nothing to write to articles.parquet."
            )
        validated = validate_articles(raw)
        assert_articles_primary_key_unique(validated)

        # Derive year/month partition keys from the tz-aware UTC published_at.
        out = validated.copy()
        out["year"] = out["published_at"].dt.year
        out["month"] = out["published_at"].dt.month

        # Full rewrite: clear stale partitions before writing.
        if out_path.exists():
            shutil.rmtree(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

        partition_cols = articles_cfg.get("partition_cols", ["year", "month"])
        out.to_parquet(
            out_path,
            partition_cols=partition_cols,
            index=False,
        )
        return out_path

    def join_articles_to_prices(
        self,
        prices_path: str | Path | None = None,
        articles_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Point-in-time join articles to prices via the XNYS session-close
        boundaries.

        Each article is bound to exactly one trading period ``P`` such that::

            period_close(P-1) < published_at <= period_close(P)

        where ``period_close`` is the NYSE session close (16:00 ET, except
        early-close sessions) from ``pandas_market_calendars``'s ``"XNYS"``
        calendar. Holiday / weekend roll-forward is implicit: the XNYS
        schedule simply omits non-trading days, so an article published on
        2024-07-04 (US holiday) or 2024-07-06 (Saturday) binds to the next
        trading session (2024-07-05 / 2024-07-08 respectively).

        The comparison is performed in **UTC**: ``period_close_ts``
        (tz-aware ``America/New_York``) is converted to UTC and
        ``published_at`` is already UTC, so the inequality is DST-safe
        (no wall-clock arithmetic across DST transitions).

        Parameters
        ----------
        prices_path:
            Path to the partitioned prices parquet store (or a single parquet
            file). When ``None``, falls back to ``[prices].output_dir`` from
            the universe TOML. The 9 canonical price columns are selected
            explicitly (``year``/``month`` Hive keys are NOT propagated).
        articles_path:
            Path to the partitioned articles parquet store (or a single
            parquet file). When ``None``, falls back to
            ``[articles].output_dir`` from the universe TOML.
        output_dir:
            Output directory for ``articles_joined.parquet``. When ``None``,
            falls back to ``[articles].output_dir``'s parent directory.

        Returns
        -------
        pathlib.Path
            The path to ``articles_joined.parquet`` (absolute).

        Raises
        ------
        ValueError
            If prices or articles are empty after the alive-ticker filter.
        """
        from pandas_market_calendars import get_calendar

        from equity.data.schema import (
            CANONICAL_ARTICLE_COLUMNS,
            CANONICAL_PRICE_COLUMNS,
            EXCHANGE_TZ,
            validate_articles,
            validate_prices,
        )

        # ---- Resolve prices ------------------------------------------------
        prices_cfg = self._load_prices_config()
        if prices_path is not None:
            prices_p = Path(prices_path)
        else:
            prices_p = Path(prices_cfg.get("output_dir", "data/equity/prices"))
        if not prices_p.is_absolute():
            prices_p = PROJECT_ROOT / prices_p
        if not prices_p.exists():
            raise FileNotFoundError(f"Prices store not found: {prices_p}")

        # ---- Resolve articles ---------------------------------------------
        articles_cfg = self._load_articles_config()
        if articles_path is not None:
            articles_p = Path(articles_path)
        else:
            articles_p = Path(
                articles_cfg.get("output_dir", "data/equity/articles")
            )
        if not articles_p.is_absolute():
            articles_p = PROJECT_ROOT / articles_p
        if not articles_p.exists():
            raise FileNotFoundError(f"Articles store not found: {articles_p}")

        # ---- Output path ---------------------------------------------------
        if output_dir is not None:
            out_dir = Path(output_dir)
        else:
            out_dir = articles_p.parent
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "articles_joined.parquet"

        # ---- Load + validate prices (select canonical columns) -----------
        prices_raw = pd.read_parquet(prices_p)
        # Select the 9 canonical columns explicitly; do NOT propagate the
        # ``year``/``month`` Hive partition keys (they are not data columns).
        missing = [c for c in CANONICAL_PRICE_COLUMNS if c not in prices_raw.columns]
        if missing:
            raise ValueError(
                f"Prices store missing canonical columns: {missing}."
            )
        prices = validate_prices(prices_raw[CANONICAL_PRICE_COLUMNS])

        # ---- Load + validate articles -------------------------------------
        articles_raw = pd.read_parquet(articles_p)
        missing = [c for c in CANONICAL_ARTICLE_COLUMNS if c not in articles_raw.columns]
        if missing:
            raise ValueError(
                f"Articles store missing canonical columns: {missing}."
            )
        articles = validate_articles(articles_raw[CANONICAL_ARTICLE_COLUMNS])

        # ---- Alive-ticker filter (delisting-aware) -----------------------
        alive = {t for (t, _l, _d) in self.universe()}
        articles = articles[articles["ticker"].isin(alive)]
        # Drop rows with NaN ticker (defensive; validate_articles already
        # rejects null tickers, but belt-and-braces for a hand-edited frame).
        articles = articles.dropna(subset=["ticker"])

        if prices.empty:
            raise ValueError("Prices frame is empty -- cannot join articles.")
        if articles.empty:
            raise ValueError(
                "Articles frame is empty after the alive-ticker filter -- "
                "nothing to join."
            )

        # ---- Build the XNYS session-close boundary index ------------------
        # Use the loader's [start, end] window (extended by 1 day on each
        # side) so articles published near the window edges still find a
        # session. Sessions outside this range are not consulted.
        cal = get_calendar("XNYS")
        sched_start = self.start - pd.Timedelta(days=1)
        sched_end = self.end + pd.Timedelta(days=1)
        sched = cal.schedule(
            sched_start.strftime("%Y-%m-%d"),
            sched_end.strftime("%Y-%m-%d"),
            tz=EXCHANGE_TZ,
        )
        # sched["market_close"] is tz-aware America/New_York. Convert to UTC
        # for the comparison; the inequality
        #   period_close(P-1) < published_at <= period_close(P)
        # is then evaluated entirely in UTC (DST-safe). We sort the closes
        # ascending and use Index.get_indexer with method="bfill", which
        # returns the smallest index i such that closes[i] >= published_at --
        # exactly the right-hand side of the rule. The left-hand side
        # (period_close(P-1) < published_at) is trivially satisfied for pos==0
        # (treat period_close(P-1) as -infinity) and, for pos>0, is implied by
        # the bfill semantics: if closes[pos-1] >= published_at then bfill would
        # have returned pos-1 instead.
        closes_utc = sched["market_close"].dt.tz_convert("UTC").sort_values()
        closes_idx = pd.DatetimeIndex(closes_utc)  # tz-aware UTC, sorted

        joined_rows: list[dict] = []
        for _, row in articles.iterrows():
            pub = pd.Timestamp(row["published_at"])
            if pub.tz is None:
                pub = pub.tz_localize("UTC")
            else:
                pub = pub.tz_convert("UTC")
            # Smallest i such that closes_idx[i] >= pub.
            pos_arr = closes_idx.get_indexer([pub], method="bfill")
            pos = int(pos_arr[0])
            if pos < 0:
                # published_at > last session close -- outside the window;
                # skip (cannot bind to a period).
                continue
            period_close = pd.Timestamp(closes_idx[pos]).tz_convert(EXCHANGE_TZ)
            joined_rows.append(
                {
                    "ticker": row["ticker"],
                    "period_close_ts": period_close,
                    "published_at": pub,
                    "text": row["text"],
                    "source": row["source"],
                }
            )

        if not joined_rows:
            raise ValueError(
                "No articles could be bound to a trading period in the "
                "loader's [start, end] window."
            )
        joined = pd.DataFrame(joined_rows)
        joined = joined[["ticker", "period_close_ts", "published_at", "text", "source"]]
        joined.to_parquet(out_file, index=False)
        return out_file


__all__ = ["EquityDataLoader"]
