"""
@module: equity.data.loader
@depends: pandas, tomllib, equity.data.registry
@exports: EquityDataLoader
@paper_ref: N/A
@data_flow: universe config -> universe_file CSV -> alive-ticker tuples
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from equity.data.fetch import STORE_MARKER
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


def _resolve_store_path(out_dir: str | Path, default_rel: str) -> Path:
    """Resolve a store output dir to an absolute path under PROJECT_ROOT.

    Relative paths are joined to ``PROJECT_ROOT``; absolute paths are honored
    verbatim (the caller is responsible for not pointing at ``C:\\`` etc. --
    :func:`_safe_rmtree` enforces a containment guard before any deletion).
    """
    out_path = Path(out_dir)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    return out_path


def _ensure_store_marker(out_path: Path) -> None:
    """Write the ``.equity_store`` sentinel marker into ``out_path`` (first
    creation). The marker is what permits :func:`_safe_rmtree` to clear the
    directory on a later rewrite -- its absence on an existing tree is
    treated as a misconfigured path and the rmtree is refused.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    marker = out_path / STORE_MARKER
    if not marker.exists():
        marker.write_text(
            "equity parquet store -- safe to rewrite via EquityDataLoader\n",
            encoding="utf-8",
        )


def _safe_rmtree(out_path: Path) -> None:
    """Remove ``out_path`` if and only if it is inside ``PROJECT_ROOT`` AND
    contains the ``.equity_store`` marker.

    Defense against a misconfigured TOML / CLI ``output_dir`` pointing at an
    existing tree outside the repo (``C:\\Users\\...``, ``~/``, repo root). A
    missing marker on an existing directory is treated as "this is not ours"
    and the rmtree is refused (raise :class:`ValueError`); on first creation
    the marker is written by :func:`_ensure_store_marker`.
    """
    resolved = out_path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise ValueError(
            f"Refusing to remove {out_path}: resolved path "
            f"{resolved} is outside PROJECT_ROOT ({PROJECT_ROOT}). "
            "Store output_dir must live under the repo root."
        )
    if not resolved.exists():
        return
    marker = resolved / STORE_MARKER
    if not marker.exists():
        raise ValueError(
            f"Refusing to remove {out_path}: missing sentinel marker "
            f"'{STORE_MARKER}' (the directory is not recognized as an "
            "equity store -- refusing to delete a non-empty tree)."
        )
    log.info("Removing equity store at %s (rewrite).", resolved)
    shutil.rmtree(resolved)


def _store_content_hash(df: pd.DataFrame) -> str:
    """Return a deterministic sha256 over the canonical price/articles frame.

    The frame is serialized to a sorted JSON list of records (columns in
    canonical order, rows sorted by all columns) so two calls producing the
    same logical frame yield the same hash regardless of row order.
    """
    cols = list(df.columns)
    sorted_df = df.sort_values(cols).reset_index(drop=True)
    payload = sorted_df.to_json(orient="records", date_format="iso", default_handler=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_store_meta(
    out_path: Path,
    *,
    kind: str,
    ticker: str | None,
    period: str,
    row_count: int,
    content_hash: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write ``_meta.json`` next to the parquet store with provenance fields.

    Records the fetch UTC timestamp, the data ``kind`` (``"prices"`` /
    ``"articles"``), the source library version (yfinance), the requested
    ``period`` / ticker, the row count and a content hash so two runs over
    the same window can be compared for reproducibility drift (see review
    finding M3 -- yfinance is an unofficial scraper whose historical values
    can be revised retroactively).
    """
    meta: dict[str, Any] = {
        "kind": kind,
        "fetched_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "row_count": int(row_count),
        "content_sha256": content_hash,
        "period": period,
        "ticker": ticker,
    }
    if extra:
        meta.update(extra)
    meta_path = out_path / "_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta_path


log = logging.getLogger(__name__)


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

    def fetch_prices(
        self,
        output_dir: str | Path | None = None,
        frozen: bool = False,
    ) -> Path:
        """Fetch OHLCV bars for the alive universe and write a partitioned
        parquet store.

        Flow:

        1. Resolve the alive-ticker set via :meth:`universe` (delisting-aware).
        2. Fetch OHLCV via :func:`equity.data.fetch.fetch_yfinance_ohlcv`
           (in-process yfinance; per-ticker errors are logged and skipped).
           The yfinance daily-bar index (00:00 ET) is canonicalized to the
           XNYS session close (16:00 / 13:00 ET) so ``period_close_ts`` is a
           valid foreign key for the S1.3 join.
        3. Validate the long-form frame with
           :func:`equity.data.schema.validate_prices` (pandera schema + tz-aware
           ``period_close_ts`` guard) and assert primary-key uniqueness via
           :func:`equity.data.schema.assert_primary_key_unique`.
        4. Write a Hive-style partitioned parquet dataset under ``output_dir``,
           partitioned by ``year``/``month`` derived from ``period_close_ts``
           (America/New_York session close).
        5. Write ``_meta.json`` next to the store with provenance (fetch UTC
           timestamp, yfinance version, row count, sha256 content hash) so two
           runs over the same window can be compared for reproducibility drift.

        The write is a **full rewrite** (not incremental): if ``output_dir``
        exists it is removed first (via :func:`_safe_rmtree`, which refuses to
        delete paths outside ``PROJECT_ROOT`` or missing the ``.equity_store``
        sentinel marker) so stale partitions from prior runs do not leak in.

        Parameters
        ----------
        output_dir:
            Output directory for the partitioned parquet store. When ``None``,
            falls back to ``[prices].output_dir`` from the universe TOML
            (default ``data/equity/prices`` relative to ``PROJECT_ROOT``).
        frozen:
            When ``True``, refuse to re-fetch if the store already exists with
            a ``_meta.json`` whose ``period`` covers ``[start, end]``. This is
            a best-effort protection against silent data drift from
            retroactively revised yfinance values (review finding M3). The
            store is left untouched; the existing ``out_path`` is returned.

        Returns
        -------
        pathlib.Path
            The output directory Path (absolute).

        Raises
        ------
        ValueError
            If the alive universe is empty for the window, or if the resolved
            ``output_dir`` is outside ``PROJECT_ROOT`` / missing the marker on
            an existing tree, or if ``frozen=True`` and the store already
            covers the window.
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
            out_path = _resolve_store_path(output_dir, "data/equity/prices")
        else:
            out_path = _resolve_store_path(
                prices_cfg.get("output_dir", "data/equity/prices"),
                "data/equity/prices",
            )

        if frozen and out_path.exists():
            meta_path = out_path / "_meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta.get("kind") == "prices" and meta.get("period") == self.period:
                        log.info(
                            "frozen prices store already present at %s "
                            "(period=%s); skipping re-fetch.",
                            out_path,
                            self.period,
                        )
                        return out_path
                except (OSError, ValueError) as exc:
                    log.warning(
                        "Could not parse %s (%s); ignoring frozen flag.",
                        meta_path,
                        exc,
                    )

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

        # Full rewrite: clear stale partitions before writing. The guard
        # refuses to rmtree paths outside PROJECT_ROOT or missing the marker.
        if out_path.exists():
            _safe_rmtree(out_path)
        _ensure_store_marker(out_path)

        partition_cols = prices_cfg.get("partition_cols", ["year", "month"])
        out.to_parquet(
            out_path,
            partition_cols=partition_cols,
            index=False,
        )

        try:
            import yfinance as _yf

            yf_version = _yf.__version__
        except Exception:  # pragma: no cover - yfinance is an optional dep
            yf_version = None
        _write_store_meta(
            out_path,
            kind="prices",
            ticker=None,
            period=self.period,
            row_count=int(len(out)),
            content_hash=_store_content_hash(validated),
            extra={
                "yfinance_version": yf_version,
                "universe": self.universe_name,
                "window_start": self.start.date().isoformat(),
                "window_end": self.end.date().isoformat(),
                "tickers_count": int(len(tickers)),
            },
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

    def fetch_articles(
        self,
        output_dir: str | Path | None = None,
        frozen: bool = False,
    ) -> Path:
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
        5. Write ``_meta.json`` next to the store with provenance (fetch UTC
           timestamp, row count, sha256 content hash).

        The write is a **full rewrite** (not incremental): if ``output_dir``
        exists it is removed first (via :func:`_safe_rmtree`) so stale
        partitions do not leak in.

        Parameters
        ----------
        output_dir:
            Output directory for the partitioned parquet store. When ``None``,
            falls back to ``[articles].output_dir`` from the universe TOML
            (default ``data/equity/articles`` relative to ``PROJECT_ROOT``).
        frozen:
            When ``True``, refuse to re-write if the store already exists with
            a ``_meta.json``. Best-effort protection against silent drift.

        Returns
        -------
        pathlib.Path
            The output directory Path (absolute).

        Raises
        ------
        ValueError
            If the seed frame is empty after validation, or if the resolved
            ``output_dir`` is outside ``PROJECT_ROOT`` / missing the marker.
        """
        from equity.data.fetch import fetch_articles_from_seed
        from equity.data.schema import (
            assert_articles_primary_key_unique,
            validate_articles,
        )

        articles_cfg = self._load_articles_config()
        if output_dir is not None:
            out_path = _resolve_store_path(output_dir, "data/equity/articles")
        else:
            out_path = _resolve_store_path(
                articles_cfg.get("output_dir", "data/equity/articles"),
                "data/equity/articles",
            )

        if frozen and out_path.exists() and (out_path / "_meta.json").exists():
            log.info(
                "frozen articles store already present at %s; skipping re-write.",
                out_path,
            )
            return out_path

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
            _safe_rmtree(out_path)
        _ensure_store_marker(out_path)

        partition_cols = articles_cfg.get("partition_cols", ["year", "month"])
        out.to_parquet(
            out_path,
            partition_cols=partition_cols,
            index=False,
        )

        _write_store_meta(
            out_path,
            kind="articles",
            ticker=None,
            period="seed",
            row_count=int(len(out)),
            content_hash=_store_content_hash(validated),
            extra={
                "universe": self.universe_name,
                "seed_file": str(seed_path),
            },
        )
        return out_path

    def join_articles_to_prices(
        self,
        prices_path: str | Path | None = None,
        articles_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> Path:
        """Point-in-time join articles to prices via the price-store session
        boundaries.

        Each article is bound to exactly one trading period ``P`` such that::

            period_close(P-1) < published_at <= period_close(P)

        where ``period_close`` is the NYSE session close (16:00 ET, except
        early-close sessions). The comparison is performed in **UTC**:
        ``period_close_ts`` (tz-aware ``America/New_York``) is converted to UTC
        and ``published_at`` is already UTC, so the inequality is DST-safe
        (no wall-clock arithmetic across DST transitions).

        Binding index
        -------------
        The bfill index is built **from the prices store's actual
        ``period_close_ts`` values** (sorted, de-duplicated, in UTC) -- NOT
        from the XNYS schedule for the loader window. This guarantees every
        bound ``period_close_ts`` is a real key in ``prices.parquet`` (no
        phantom FK). Articles published after the last stored session close
        are **dropped** (no +1-day extension: an article published Friday
        17:00 ET binds to the next session *present in prices.parquet within
        the window*; if none exists, the article is dropped -- under-inclusion,
        NOT leakage; see README "Under-inclusion at window edges").

        Both ``period_close_ts`` and ``published_at`` are written to
        ``articles_joined.parquet`` canonicalized to **UTC** (downstream S2+
        slices can compare them without tz juggling).

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
        FileNotFoundError
            If the prices or articles store does not exist.
        ValueError
            If prices or articles are empty after the alive-ticker filter, or
            if no articles could be bound to a stored session.
        """
        from equity.data.schema import (
            CANONICAL_ARTICLE_COLUMNS,
            CANONICAL_PRICE_COLUMNS,
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

        # ---- Build the bfill index from the ACTUAL prices sessions -------
        # Restrict to the price sessions that actually exist in the store so
        # every bound period_close_ts is a valid FK into prices.parquet (no
        # phantom keys from a schedule-derived window). Drop the loader's
        # [start-1d, end+1d] schedule entirely (review finding B2).
        closes_utc = pd.DatetimeIndex(
            sorted(set(prices["period_close_ts"].dt.tz_convert("UTC").unique()))
        )
        if len(closes_utc) == 0:
            raise ValueError("Prices store has no session-close timestamps.")

        # Vectorized bfill (review finding m3): single get_indexer call on the
        # numpy array, no Python per-row loop. bfill returns the smallest i
        # such that closes_utc[i] >= published_at -- the right-hand side of
        # the PIT rule. -1 means published_at > last stored session close
        # (article outside the store's coverage -> drop, under-inclusion).
        pub_utc = articles["published_at"].dt.tz_convert("UTC").to_numpy()
        pos = closes_utc.get_indexer(pub_utc, method="bfill")

        joined_rows: list[dict] = []
        n_dropped = 0
        for i, p in enumerate(pos):
            if p < 0:
                n_dropped += 1
                continue
            period_close_utc = pd.Timestamp(closes_utc[p])
            row = articles.iloc[i]
            joined_rows.append(
                {
                    "ticker": row["ticker"],
                    # Canonicalize BOTH columns to UTC in the joined output
                    # (review finding n3) so downstream slices are tz-uniform.
                    "period_close_ts": period_close_utc,
                    "published_at": pd.Timestamp(pub_utc[i]),
                    "text": row["text"],
                    "source": row["source"],
                }
            )

        if not joined_rows:
            raise ValueError(
                "No articles could be bound to a trading period present in "
                "the prices store (all " + str(len(articles)) + " articles "
                "were published after the last stored session close)."
            )
        if n_dropped > 0:
            log.info(
                "join_articles_to_prices: dropped %d article(s) published "
                "after the last stored session close (under-inclusion).",
                n_dropped,
            )

        joined = pd.DataFrame(joined_rows)
        joined = joined[["ticker", "period_close_ts", "published_at", "text", "source"]]
        # Canonicalize both timestamp columns to UTC in the parquet output.
        joined["period_close_ts"] = joined["period_close_ts"].astype(
            pd.DatetimeTZDtype(tz="UTC")
        )
        joined["published_at"] = joined["published_at"].astype(
            pd.DatetimeTZDtype(tz="UTC")
        )
        joined.to_parquet(out_file, index=False)
        return out_file


__all__ = ["EquityDataLoader"]
