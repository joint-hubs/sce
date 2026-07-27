"""
@module: equity.sentiment.aggregate
@depends: pandas, numpy, pandas_market_calendars
@exports: aggregate_per_period, aggregate_market_wide, write_sentiment_per_period,
          write_market_wide
@paper_ref: N/A
@data_flow: per-article scores -> time-decayed per-(ticker,period) aggregate +
            market-wide aggregate

Time-decayed per-(ticker, period) sentiment aggregation.

Formula
-------
For each trading period ``P`` and each article ``t`` published before
``period_close_P`` (point-in-time safe), the time-decayed weight is::

    w_t = exp(-dt_t / halflife)

where ``dt_t = (period_close_P - published_at_t)`` converted to days and
``halflife`` is :attr:`sentiment_halflife_days` (default 5). The aggregate
score for period ``P`` (and likewise ``pos_P``, ``neg_P``, ``neu_P``) is the
weighted mean::

    sentiment_score_P = sum(w_t * score_t) / sum(w_t)
    n_articles_P      = count(t)

A smaller ``halflife`` discounts older articles faster; ``halflife -> inf``
converges to the simple mean.

PIT safety
----------
The aggregator only consumes articles with ``published_at <= period_close_P``
(bound to period ``P`` by S1.3's PIT rule). The guard
:mod:`equity.diagnostics.sentiment_aggregate_guard` asserts this invariant
explicitly on the output.

Market-wide aggregate
---------------------
Articles whose ticker is NOT in the S1 alive set (e.g. the
``__TEST_NOT_IN_UNIVERSE__`` sentinel, or a pre-window LEH article dropped
by S1's alive filter) roll into a separate ``market_wide_sentiment.parquet``
(cols: ``period_close_ts, sentiment_score, sentiment_pos, sentiment_neg,
sentiment_neu, n_articles``; no ``ticker`` col). Source: raw
``articles.parquet`` (NOT ``articles_joined.parquet`` -- the latter is
already filtered to alive tickers).

Path taken (FOC-49 decision): the FAITHFUL path -- raw ``articles.parquet``
is assigned to a period via the XNYS session-close calendar reused from S1.
The calendar helper lives inside :func:`equity.data.fetch._canonicalize_session_close`
which is private to that module; rather than import a private helper, we
rebuild the XNYS schedule via ``pandas_market_calendars.get_calendar("XNYS")``
with the same window-extent logic the loader uses, and apply the SAME PIT
rule (``period_close(P-1) < published_at <= period_close(P)``) via
``get_indexer(method="bfill")`` on the sorted XNYS closes. This mirrors
:meth:`EquityDataLoader.join_articles_to_prices` exactly, so per-ticker and
market-wide aggregates use the same period assignment.

If ``pandas_market_calendars`` is unavailable (e.g. minimal install), the
market-wide aggregator falls back to grouping ``articles_joined.parquet``
by ``period_close_ts`` (pooling all resolved tickers) -- documented in the
caller's docstring. This fallback LOSES the unresolved-ticker articles but
keeps the rest of the pipeline functional.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equity.data.registry import PROJECT_ROOT
from equity.sentiment.schema import (
    CANONICAL_MARKET_WIDE_COLUMNS,
    CANONICAL_PER_PERIOD_COLUMNS,
    SENTIMENT_TZ,
    assert_market_wide_primary_key_unique,
    assert_per_period_primary_key_unique,
    validate_market_wide,
    validate_sentiment_per_period,
)

log = logging.getLogger(__name__)

DEFAULT_HALFLIFE_DAYS = 5.0


def _build_xnys_schedule(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Build a sorted UTC index of XNYS session-close timestamps covering
    ``[start, end]``. Mirrors :func:`equity.data.fetch._canonicalize_session_close`
    but returns the raw closes (not a per-row mapping).

    The XNYS calendar is the same one used in S1. We expand the window by
    +1 day on each side to absorb edge articles published just outside the
    requested window (mirrors the loader's +1d ``end`` offset).
    """
    from pandas_market_calendars import get_calendar

    cal = get_calendar("XNYS")
    start_str = (start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    end_str = (end + pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    schedule = cal.schedule(start_str, end_str, tz="America/New_York")
    closes = pd.DatetimeIndex(schedule["market_close"]).tz_convert(SENTIMENT_TZ)
    return pd.DatetimeIndex(sorted(closes))


def _assign_period_via_bfill(
    published_at: pd.DatetimeIndex, closes_utc: pd.DatetimeIndex
) -> np.ndarray:
    """Return the index into ``closes_utc`` for each ``published_at`` via
    bfill (smallest ``i`` such that ``closes_utc[i] >= published_at``).

    Mirrors :meth:`EquityDataLoader.join_articles_to_prices` PIT rule
    ``period_close(P-1) < published_at <= period_close(P)``. Returns ``-1``
    for articles published after the last stored session close (these are
    dropped by the caller -- under-inclusion, NOT leakage).
    """
    pos = closes_utc.get_indexer(published_at, method="bfill")
    return pos


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Return ``sum(w * v) / sum(w)`` (or 0.0 when ``sum(w) == 0``)."""
    total_w = float(weights.sum())
    if total_w == 0.0:
        return 0.0
    return float((weights * values).sum() / total_w)


def _compute_weights(
    period_close_utc: pd.Timestamp, published_at_utc: pd.Series, halflife_days: float
) -> np.ndarray:
    """Compute ``w_t = exp(-dt_t / halflife)`` where
    ``dt_t = (period_close_P - published_at_t)`` in days.

    All inputs are tz-aware UTC. Returns a float64 numpy array aligned with
    ``published_at_utc``.
    """
    if halflife_days <= 0:
        raise ValueError(f"halflife_days must be > 0 (got {halflife_days}).")
    dt_seconds = (period_close_utc - published_at_utc).dt.total_seconds().to_numpy()
    # Negative dt would mean published_at > period_close (PIT violation).
    # The aggregator drops such rows upstream; assert here as defense.
    if (dt_seconds < 0).any():
        raise ValueError(
            "PIT violation: at least one article has published_at > "
            "period_close_ts. The aggregator must drop these upstream."
        )
    dt_days = dt_seconds / 86400.0
    return np.exp(-dt_days / float(halflife_days))


def _aggregate_group(
    group: pd.DataFrame, period_close: pd.Timestamp, halflife_days: float
) -> dict[str, float | int]:
    """Compute the weighted-mean aggregate for one ``(ticker, period_close_ts)``
    group. ``group`` must contain ``published_at``, ``pos``, ``neg``,
    ``neu``, ``score`` columns. ``period_close`` is the group's
    ``period_close_ts`` (passed explicitly to avoid relying on
    ``DataFrame.name``, which is not available in pandas 2.x).
    """
    # published_at is tz-aware UTC (validated upstream); convert to UTC for
    # the dt computation.
    pub = group["published_at"].dt.tz_convert(SENTIMENT_TZ)
    weights = _compute_weights(period_close, pub, halflife_days)
    return {
        "sentiment_score": _weighted_mean(group["score"].to_numpy(), weights),
        "sentiment_pos": _weighted_mean(group["pos"].to_numpy(), weights),
        "sentiment_neg": _weighted_mean(group["neg"].to_numpy(), weights),
        "sentiment_neu": _weighted_mean(group["neu"].to_numpy(), weights),
        "n_articles": int(len(group)),
    }


def aggregate_per_period(
    per_article: pd.DataFrame, halflife_days: float = DEFAULT_HALFLIFE_DAYS
) -> pd.DataFrame:
    """Compute the time-decayed per-(ticker, period_close_ts) aggregate.

    Parameters
    ----------
    per_article:
        Canonical per-article frame (output of
        :meth:`SentimentCache.score_articles`). Must have columns
        ``ticker, published_at, pos, neg, neu, score``. ``ticker`` must be
        non-null for every row (market-wide articles go to
        :func:`aggregate_market_wide`). The frame MUST also carry a
        ``period_close_ts`` column (the S1.3 join key from
        ``articles_joined.parquet``) -- if absent, the caller must merge it
        in (see :func:`aggregate_from_joined`).
    halflife_days:
        Time-decay halflife in days (default 5). ``w_t = exp(-dt / halflife)``.

    Returns
    -------
    pd.DataFrame
        Canonical per-period frame (see
        :data:`CANONICAL_PER_PERIOD_COLUMNS`), validated against
        :data:`sentiment_per_period_schema`.

    Raises
    ------
    ValueError
        If ``per_article`` lacks the required columns, or if any row has a
        null ``ticker`` (market-wide articles must be routed to
        :func:`aggregate_market_wide`), or if a PIT violation is detected
        (``published_at > period_close_ts``).
    """
    required = {"ticker", "published_at", "period_close_ts", "pos", "neg", "neu", "score"}
    missing = required - set(per_article.columns)
    if missing:
        raise ValueError(
            f"per_article missing required columns: {sorted(missing)}. "
            "Did you forget to merge period_close_ts from articles_joined?"
        )
    if per_article.empty:
        return pd.DataFrame(columns=CANONICAL_PER_PERIOD_COLUMNS)

    if per_article["ticker"].isna().any():
        raise ValueError(
            "per_article has null ticker values; route those rows to "
            "aggregate_market_wide instead."
        )

    df = per_article.copy()
    df["period_close_ts"] = df["period_close_ts"].dt.tz_convert(SENTIMENT_TZ)
    df["published_at"] = df["published_at"].dt.tz_convert(SENTIMENT_TZ)

    # PIT safety: drop any row with published_at > period_close_ts.
    pit_violation = df["published_at"] > df["period_close_ts"]
    if pit_violation.any():
        bad = df.loc[pit_violation, ["ticker", "published_at", "period_close_ts"]].head(5)
        raise ValueError(
            f"PIT violation: {int(pit_violation.sum())} article(s) with "
            f"published_at > period_close_ts. First 5:\n{bad.to_string(index=False)}"
        )

    rows: list[dict[str, Any]] = []
    for (ticker, period_close), group in df.groupby(["ticker", "period_close_ts"], sort=True):
        agg = _aggregate_group(group, pd.Timestamp(period_close), halflife_days)
        rows.append(
            {
                "ticker": ticker,
                "period_close_ts": period_close,
                "sentiment_score": agg["sentiment_score"],
                "sentiment_pos": agg["sentiment_pos"],
                "sentiment_neg": agg["sentiment_neg"],
                "sentiment_neu": agg["sentiment_neu"],
                "n_articles": agg["n_articles"],
            }
        )

    out = pd.DataFrame(rows, columns=CANONICAL_PER_PERIOD_COLUMNS)
    out["period_close_ts"] = out["period_close_ts"].astype(pd.DatetimeTZDtype(tz=SENTIMENT_TZ))
    out = validate_sentiment_per_period(out)
    assert_per_period_primary_key_unique(out)
    return out


def aggregate_market_wide(
    per_article: pd.DataFrame, halflife_days: float = DEFAULT_HALFLIFE_DAYS
) -> pd.DataFrame:
    """Compute the time-decayed market-wide aggregate (no ticker col).

    ``per_article`` here is the market-wide per-article frame (output of
    :func:`build_market_wide_per_article`): one row per article, with
    ``period_close_ts`` already assigned via the XNYS calendar. ``ticker``
    is ignored (these are unresolved-ticker articles).

    Returns
    -------
    pd.DataFrame
        Canonical market-wide frame (see :data:`CANONICAL_MARKET_WIDE_COLUMNS`).
    """
    required = {"published_at", "period_close_ts", "pos", "neg", "neu", "score"}
    missing = required - set(per_article.columns)
    if missing:
        raise ValueError(f"per_article missing required columns: {sorted(missing)}.")
    if per_article.empty:
        return pd.DataFrame(columns=CANONICAL_MARKET_WIDE_COLUMNS)

    df = per_article.copy()
    df["period_close_ts"] = df["period_close_ts"].dt.tz_convert(SENTIMENT_TZ)
    df["published_at"] = df["published_at"].dt.tz_convert(SENTIMENT_TZ)

    pit_violation = df["published_at"] > df["period_close_ts"]
    if pit_violation.any():
        bad = df.loc[pit_violation, ["published_at", "period_close_ts"]].head(5)
        raise ValueError(
            f"PIT violation: {int(pit_violation.sum())} article(s) with "
            f"published_at > period_close_ts. First 5:\n{bad.to_string(index=False)}"
        )

    rows: list[dict[str, Any]] = []
    for period_close, group in df.groupby("period_close_ts", sort=True):
        # Build a single-row "group" representation compatible with
        # _aggregate_group (which expects a tuple name).
        weights = _compute_weights(pd.Timestamp(period_close), group["published_at"], halflife_days)
        rows.append(
            {
                "period_close_ts": period_close,
                "sentiment_score": _weighted_mean(group["score"].to_numpy(), weights),
                "sentiment_pos": _weighted_mean(group["pos"].to_numpy(), weights),
                "sentiment_neg": _weighted_mean(group["neg"].to_numpy(), weights),
                "sentiment_neu": _weighted_mean(group["neu"].to_numpy(), weights),
                "n_articles": int(len(group)),
            }
        )

    out = pd.DataFrame(rows, columns=CANONICAL_MARKET_WIDE_COLUMNS)
    out["period_close_ts"] = out["period_close_ts"].astype(pd.DatetimeTZDtype(tz=SENTIMENT_TZ))
    out = validate_market_wide(out)
    assert_market_wide_primary_key_unique(out)
    return out


# ---------------------------------------------------------------------------
# Convenience: build the market-wide per-article frame from raw articles.parquet
# ---------------------------------------------------------------------------


def build_market_wide_per_article(
    articles: pd.DataFrame,
    *,
    window_start: pd.Timestamp | None = None,
    window_end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Assign a ``period_close_ts`` to each row of raw ``articles.parquet``
    via the XNYS session-close calendar (faithful path).

    Parameters
    ----------
    articles:
        Raw ``articles.parquet`` frame (cols ``ticker, published_at, text,
        source``). ``published_at`` must be tz-aware UTC.
    window_start, window_end:
        Optional window bounds. When provided, the XNYS schedule is built
        for ``[window_start - 1d, window_end + 2d]`` (mirrors the loader's
        +1d end offset). When omitted, the schedule is derived from the
        articles' published_at min/max.

    Returns
    -------
    pd.DataFrame
        Articles frame with an added ``period_close_ts`` column (tz-aware
        UTC). Articles published before the first XNYS session or after the
        last XNYS session in the schedule are DROPPED (under-inclusion,
        NOT leakage -- mirrors S1.3 ``join_articles_to_prices``).
    """
    if "published_at" not in articles.columns:
        raise ValueError("articles frame missing required column 'published_at'.")
    if articles.empty:
        out = articles.copy()
        out["period_close_ts"] = pd.Series(dtype=pd.DatetimeTZDtype(tz=SENTIMENT_TZ))
        return out

    pub = articles["published_at"].dt.tz_convert(SENTIMENT_TZ)
    if window_start is None:
        window_start = pub.min()
    if window_end is None:
        window_end = pub.max()
    closes_utc = _build_xnys_schedule(pd.Timestamp(window_start), pd.Timestamp(window_end))
    pos = _assign_period_via_bfill(pd.DatetimeIndex(pub.to_numpy()), closes_utc)

    first_close = closes_utc[0]
    last_close = closes_utc[-1]
    # PIT rule via bfill: ``period_close(P-1) < published_at <= period_close(P)``.
    # An article published BEFORE the first stored session close (``pub <
    # first_close``) still binds to the first session (LHS vacuous -- there
    # is no P-1). This DIVERGES from S1.3's per-ticker
    # ``join_articles_to_prices`` which DROPS pre-first-session articles
    # (q7: no prior_close to form a PIT training pair). The market-wide
    # aggregate is NOT forming PIT training pairs -- it just computes a
    # per-period weighted mean -- so binding stale weekend/holiday articles
    # to the next session close is the correct PIT semantics here.
    #
    # Articles published AFTER the last stored session close (``pos == -1``)
    # are dropped (under-inclusion, NOT leakage -- mirrors S1.3).
    pos_series = pd.Series(pos, index=articles.index)
    keep_mask = pos_series >= 0
    kept = articles.loc[keep_mask].copy()
    kept_pos = pos[keep_mask.to_numpy()]
    kept["period_close_ts"] = [closes_utc[i] for i in kept_pos]
    # ``first_close`` / ``last_close`` referenced for documentation above;
    # silence unused-variable lint while keeping the explanatory locals.
    _ = first_close, last_close
    kept["period_close_ts"] = kept["period_close_ts"].astype(pd.DatetimeTZDtype(tz=SENTIMENT_TZ))
    kept["published_at"] = kept["published_at"].dt.tz_convert(SENTIMENT_TZ)
    return kept.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def _resolve_under_project_root(path: str | Path, default_rel: str) -> Path:
    out_path = Path(path) if path else Path(default_rel)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    resolved = out_path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise ValueError(f"Refusing to write outside PROJECT_ROOT: {resolved}.")
    return resolved


def write_sentiment_per_period(per_period: pd.DataFrame, out_path: str | Path) -> Path:
    """Write ``sentiment_per_period.parquet`` (Hive-partitioned year/month
    by ``period_close_ts`` UTC) mirroring S1's parquet-store pattern.
    """
    validated = validate_sentiment_per_period(per_period)
    assert_per_period_primary_key_unique(validated)
    out = _resolve_under_project_root(out_path, "data/equity/sentiment_per_period")
    out.mkdir(parents=True, exist_ok=True)
    df = validated.copy()
    df["year"] = df["period_close_ts"].dt.year
    df["month"] = df["period_close_ts"].dt.month
    df.to_parquet(out, partition_cols=["year", "month"], index=False)
    return out


def write_market_wide(market_wide: pd.DataFrame, out_path: str | Path) -> Path:
    """Write ``market_wide_sentiment.parquet`` (Hive-partitioned year/month
    by ``period_close_ts`` UTC).
    """
    validated = validate_market_wide(market_wide)
    assert_market_wide_primary_key_unique(validated)
    out = _resolve_under_project_root(out_path, "data/equity/market_wide_sentiment")
    out.mkdir(parents=True, exist_ok=True)
    df = validated.copy()
    df["year"] = df["period_close_ts"].dt.year
    df["month"] = df["period_close_ts"].dt.month
    df.to_parquet(out, partition_cols=["year", "month"], index=False)
    return out


__all__ = [
    "DEFAULT_HALFLIFE_DAYS",
    "aggregate_per_period",
    "aggregate_market_wide",
    "build_market_wide_per_article",
    "write_sentiment_per_period",
    "write_market_wide",
]
