"""
@module: equity.sentiment.aggregate
@depends: pandas, numpy, pandas_market_calendars
@exports: aggregate_per_period, aggregate_market_wide, aggregate_from_joined,
          write_sentiment_per_period, write_market_wide
@paper_ref: N/A
@data_flow: per-article scores -> time-decayed per-(ticker,period) aggregate +
            market-wide aggregate

Time-decayed per-(ticker, period) sentiment aggregation.

Formula
-------
For each trading period ``P`` and each article ``t`` published before
``period_close_P`` (point-in-time safe), the time-decayed weight is::

    w_t = exp(-dt_t / tau)

where ``dt_t = (period_close_P - published_at_t)`` converted to days and
``tau`` is :data:`DEFAULT_DECAY_TIME_CONST_DAYS` (default 5). The aggregate
score for period ``P`` (and likewise ``pos_P``, ``neg_P``, ``neu_P``) is the
weighted mean::

    sentiment_score_P = sum(w_t * score_t) / sum(w_t)
    n_articles_P      = count(t)

A smaller ``tau`` discounts older articles faster; ``tau -> inf``
converges to the simple mean.

Note on the ``tau`` naming (FOC-49 M2)
---------------------------------------
The parameter was renamed from ``halflife_days`` to
``decay_time_const_days`` for mathematical honesty: the formula
``w = exp(-dt / tau)`` makes ``tau`` the EXPONENTIAL TIME CONSTANT (the
``1/e`` folding time), NOT the halflife (``ln(2) * tau``). The numerical
value (5) and the formula are UNCHANGED -- only the name was misleading.

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

``pandas_market_calendars`` is a HARD dependency for the market-wide
aggregation path (used by :func:`build_market_wide_per_article` /
:func:`_build_xnys_schedule` to assign each article to an XNYS session
close). It is NOT needed for the per-ticker path
(:func:`aggregate_per_period`), which consumes ``period_close_ts`` already
assigned by S1.3's ``articles_joined.parquet``. A minimal install without
``pandas_market_calendars`` will raise ``ImportError`` at first call to
:func:`_build_xnys_schedule`; install the package (it is listed in the
project's runtime dependencies) to enable the market-wide path. A
hand-rolled fallback XNYS schedule is intentionally NOT provided -- the
NYSE session logic (half-days, holidays, early closes) is non-trivial and
an incorrect fallback would introduce its own PIT bugs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from equity.sentiment.base import SentimentScorer
    from equity.sentiment.cache import SentimentCache

log = logging.getLogger(__name__)

# Exponential time constant tau for ``w = exp(-dt / tau)`` (days). The
# numerical value (5) and the formula are unchanged from S2.3; only the
# name was corrected from the misleading ``halflife_days`` (FOC-49 M2).
DEFAULT_DECAY_TIME_CONST_DAYS = 5.0


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
    """Return ``sum(w * v) / sum(w)`` (or 0.0 when ``sum(w) == 0``).

    Raises ``ValueError`` if ``values`` contains NaN -- a NaN-producing
    scorer would otherwise silently propagate NaN through the weighted mean,
    bypassing the comparison-based guard invariants (FOC-49 round-3 review
    blocker). The schema's ``nullable=False`` on probability/score columns
    is the primary defense; this is defense-in-depth.
    """
    total_w = float(weights.sum())
    if total_w == 0.0:
        return 0.0
    if np.isnan(values).any():
        raise ValueError(
            "NaN in values passed to _weighted_mean; the aggregator must "
            "reject NaN scores upstream via schema validation "
            "(nullable=False on probability/score columns)."
        )
    return float((weights * values).sum() / total_w)


def _compute_weights(
    period_close_utc: pd.Timestamp,
    published_at_utc: pd.Series,
    decay_time_const_days: float,
) -> np.ndarray:
    """Compute ``w_t = exp(-dt_t / tau)`` where
    ``dt_t = (period_close_P - published_at_t)`` in days and
    ``tau = decay_time_const_days`` is the exponential time constant
    (``1/e`` folding time, NOT the halflife which is ``ln(2)*tau``).

    All inputs are tz-aware UTC. Returns a float64 numpy array aligned with
    ``published_at_utc``.
    """
    if decay_time_const_days <= 0:
        raise ValueError(f"decay_time_const_days must be > 0 (got {decay_time_const_days}).")
    dt_seconds = (period_close_utc - published_at_utc).dt.total_seconds().to_numpy()
    # Negative dt would mean published_at > period_close (PIT violation).
    # The aggregator drops such rows upstream; assert here as defense.
    if (dt_seconds < 0).any():
        raise ValueError(
            "PIT violation: at least one article has published_at > "
            "period_close_ts. The aggregator must drop these upstream."
        )
    dt_days = dt_seconds / 86400.0
    return np.exp(-dt_days / float(decay_time_const_days))


def _aggregate_period_only(
    group: pd.DataFrame,
    period_close: pd.Timestamp,
    decay_time_const_days: float,
    *,
    drop_zero_weight: bool = False,
) -> dict[str, float | int] | None:
    """Compute the weighted-mean aggregate for ONE period's group.

    Shared by per-ticker aggregation (:func:`aggregate_per_period`) and
    market-wide aggregation (:func:`aggregate_market_wide`). ``group`` must
    contain ``published_at, pos, neg, neu, score``. ``period_close`` is
    the group's ``period_close_ts`` (passed explicitly -- groupby.name is
    unreliable across pandas versions). ``published_at`` must already be
    tz-converted to :data:`SENTIMENT_TZ` by the caller.

    When ``drop_zero_weight`` is True, rows whose weight rounds to ~0
    (``w <= 1e-6``) are excluded from BOTH the weighted mean and the
    ``n_articles`` count. Used by the market-wide aggregator (FOC-49 Q2):
    very-old articles with ``w ~= 0`` would otherwise inflate
    ``n_articles`` without contributing to the score, misrepresenting the
    period's effective sample size.

    Returns ``None`` when ``drop_zero_weight`` is True AND every row in the
    group has weight ``<= 1e-6`` (i.e. ``n_articles == 0`` after the
    zero-weight mask). In that case the period has NO contributing articles,
    so the market-wide aggregator must SKIP emitting a row (an all-zero
    0/0/0 row would trip the prob-sum invariant with a misleading error --
    FOC-49 round-3 review fix). The per-ticker path never sets
    ``drop_zero_weight`` and thus never returns ``None``.
    """
    weights = _compute_weights(period_close, group["published_at"], decay_time_const_days)
    if drop_zero_weight:
        mask = weights > 1e-6
        n_articles = int(mask.sum())
        if n_articles == 0:
            return None

        def wm(values: np.ndarray) -> float:
            return _weighted_mean(np.asarray(values)[mask], weights[mask])
    else:
        n_articles = int(len(group))

        def wm(values: np.ndarray) -> float:
            return _weighted_mean(np.asarray(values), weights)

    return {
        "sentiment_score": wm(group["score"].to_numpy()),
        "sentiment_pos": wm(group["pos"].to_numpy()),
        "sentiment_neg": wm(group["neg"].to_numpy()),
        "sentiment_neu": wm(group["neu"].to_numpy()),
        "n_articles": n_articles,
    }


def _aggregate_group(
    group: pd.DataFrame, period_close: pd.Timestamp, decay_time_const_days: float
) -> dict[str, float | int]:
    """Compute the weighted-mean aggregate for one ``(ticker, period_close_ts)``
    group. ``group`` must contain ``published_at``, ``pos``, ``neg``,
    ``neu``, ``score`` columns. ``period_close`` is the group's
    ``period_close_ts`` (passed explicitly -- the groupby ``name`` tuple is
    unreliable to depend on across pandas versions, and ``DataFrame.name``
    is not set on a plain DataFrame).
    """
    return _aggregate_period_only(group, period_close, decay_time_const_days)


def aggregate_per_period(
    per_article: pd.DataFrame,
    decay_time_const_days: float = DEFAULT_DECAY_TIME_CONST_DAYS,
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
    decay_time_const_days:
        Exponential time constant tau (days) for ``w_t = exp(-dt / tau)``.
        Default 5. Note: this is the ``1/e`` folding time, NOT the halflife
        (``ln(2) * tau``).

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
            "per_article has null ticker values; route those rows to aggregate_market_wide instead."
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
    # groupby yields the (ticker, period_close_ts) tuple as ``name``;
    # period_close is already a tz-aware Timestamp here (the groupby key
    # is a DatetimeTZDtype column), so no re-wrapping is needed.
    for (ticker, period_close), group in df.groupby(["ticker", "period_close_ts"], sort=True):
        agg = _aggregate_group(group, period_close, decay_time_const_days)
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
    per_article: pd.DataFrame,
    decay_time_const_days: float = DEFAULT_DECAY_TIME_CONST_DAYS,
) -> pd.DataFrame:
    """Compute the time-decayed market-wide aggregate (no ticker col).

    ``per_article`` here is the market-wide per-article frame (output of
    :func:`build_market_wide_per_article`): one row per article, with
    ``period_close_ts`` already assigned via the XNYS calendar. ``ticker``
    is ignored (these are unresolved-ticker articles).

    ``n_articles`` counts ONLY rows whose weight exceeds ``1e-6``
    (FOC-49 Q2): very-old articles with ``w ~= 0`` do not contribute to
    the weighted mean, so counting them in ``n_articles`` would overstate
    the period's effective sample size. The score/pos/neg/neu columns are
    likewise computed over only the contributing rows.

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
    # groupby yields the period_close_ts Timestamp as ``name``; it is
    # already tz-aware, so no re-wrapping via pd.Timestamp is needed.
    for period_close, group in df.groupby("period_close_ts", sort=True):
        agg = _aggregate_period_only(
            group, period_close, decay_time_const_days, drop_zero_weight=True
        )
        # When every article in this period has w <= 1e-6, the aggregate is
        # empty -> skip emitting a row (FOC-49 round-3 review fix). An
        # all-zero 0/0/0 row would trip the prob-sum invariant with a
        # misleading error.
        if agg is None:
            continue
        rows.append(
            {
                "period_close_ts": period_close,
                "sentiment_score": agg["sentiment_score"],
                "sentiment_pos": agg["sentiment_pos"],
                "sentiment_neg": agg["sentiment_neg"],
                "sentiment_neu": agg["sentiment_neu"],
                "n_articles": agg["n_articles"],
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
# Glue: cache -> aggregate (FOC-49 B1)
# ---------------------------------------------------------------------------


def aggregate_from_joined(
    articles_joined: pd.DataFrame,
    cache: "SentimentCache",
    scorer: "SentimentScorer",
    *,
    decay_time_const_days: float = DEFAULT_DECAY_TIME_CONST_DAYS,
    text_col: str = "text",
) -> pd.DataFrame:
    """Score articles via the cache then aggregate per (ticker, period).

    Glue function (FOC-49 B1): runs
    :meth:`SentimentCache.score_articles` on ``articles_joined`` and feeds
    the result (which carries pass-through ``ticker``, ``published_at``,
    ``period_close_ts`` from the input) into :func:`aggregate_per_period`.

    Parameters
    ----------
    articles_joined:
        DataFrame with columns ``ticker, published_at (UTC tz-aware),
        period_close_ts (UTC tz-aware, >= published_at), text, source``
        (source optional). Typically the S1.3 ``articles_joined.parquet``.
    cache:
        A :class:`SentimentCache` (warmed or cold).
    scorer:
        A :class:`SentimentScorer`.
    decay_time_const_days:
        Exponential time constant tau (days) for the time decay.
    text_col:
        Column name holding the article text (default ``"text"``).

    Returns
    -------
    pd.DataFrame
        Per-(ticker, period) aggregate (same schema as
        :func:`aggregate_per_period`).

    Raises
    ------
    ValueError
        If ``articles_joined`` lacks ``period_close_ts`` or the required
        scoring columns, or if a PIT violation is detected downstream.
    """
    if "period_close_ts" not in articles_joined.columns:
        raise ValueError(
            "articles_joined missing required column 'period_close_ts'. "
            "Use articles_joined.parquet (S1.3 join key), not raw articles.parquet."
        )
    per_article = cache.score_articles(articles_joined, scorer, text_col=text_col)
    return aggregate_per_period(per_article, decay_time_const_days=decay_time_const_days)


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
    "DEFAULT_DECAY_TIME_CONST_DAYS",
    "aggregate_per_period",
    "aggregate_market_wide",
    "aggregate_from_joined",
    "build_market_wide_per_article",
    "write_sentiment_per_period",
    "write_market_wide",
]
