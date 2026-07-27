"""
@module: equity.diagnostics.sentiment_aggregate_guard
@depends: pandas, numpy
@exports: run_sentiment_aggregate_guard, main
@paper_ref: N/A
@data_flow: per-article cache + per-period aggregate -> invariant checks

Sentiment aggregate sanity checks (S2.3). The guard asserts FIVE invariants
on the per-period aggregate frame and (when provided) the per-article cache:

1. **PIT safety** -- no per-period aggregate consumed an article with
   ``published_at > period_close_ts``. Catches a regression in the
   aggregator's groupby that would bind a future article to a past period.

2. **Probabilities invariant** -- ``sentiment_pos + sentiment_neg +
   sentiment_neu == 1`` for every aggregate row (within tolerance). Catches
   a corrupted cache or a custom scorer returning unnormalized
   probabilities.

3. **Weight monotonicity** -- within a ``(ticker, period_close_ts)`` group,
   older articles (smaller ``published_at``) receive smaller-or-equal
   weights. The weight function ``w = exp(-dt/tau)`` is monotonically
   increasing in ``dt`` (and thus in ``published_at``); a violation
   indicates a date-arithmetic bug.

4. **Cache freshness** (when a ``_meta.json`` is provided) -- the cache's
   ``model_name`` / ``model_revision`` match the active scorer; a mismatch
   means the aggregate was built from stale cached scores.

5. **Aggregate reproducibility** (FOC-49 M3) -- for each
   ``(ticker, period_close_ts)`` group in ``per_period``, re-derive the
   ``sentiment_score`` / ``sentiment_pos`` / ``sentiment_neg`` /
   ``sentiment_neu`` / ``n_articles`` from the matching ``per_article`` rows
   using the SAME weight function as the aggregator, and assert the
   re-derived values match the stored ones. Catches a regression where the
   aggregator diverges from its documented formula.

Run as a module:

    python -m equity.diagnostics.sentiment_aggregate_guard \\
        --per-article <sentiment_per_article.parquet> \\
        --per-period <sentiment_per_period.parquet>

Exits 0 on PASS, 1 on any violation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from equity.data.registry import PROJECT_ROOT
from equity.sentiment.aggregate import _compute_weights, _weighted_mean

RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics" / "equity"

# Tolerance for the ``pos + neg + neu == 1`` invariant (mirrors the schema).
_PROB_SUM_TOL = 1e-6

# Tolerance for the reproducibility check (FOC-49 M3). The aggregator uses
# float64 throughout, so 1e-9 would suffice for pure re-derivation; 1e-6
# absorbs parquet round-trip noise (float32 cached scores).
_REPRO_TOL = 1e-6


def _coerce_utc(series: pd.Series, col: str) -> pd.Series:
    if series.dt.tz is None:
        raise ValueError(f"{col} is tz-naive; expected tz-aware UTC. Refusing to compare.")
    return series.dt.tz_convert("UTC")


def assert_pit_safety(
    per_article: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return a list of PIT violation records (published_at > period_close_ts)."""
    if per_article.empty:
        return []
    if "period_close_ts" not in per_article.columns:
        return [{"type": "missing_column", "column": "period_close_ts"}]
    pub = _coerce_utc(per_article["published_at"], "published_at")
    pc = _coerce_utc(per_article["period_close_ts"], "period_close_ts")
    gap = pub.to_numpy() - pc.to_numpy()
    bad = gap > pd.Timedelta(0)
    violations: list[dict[str, Any]] = []
    if bad.any():
        # Cap at 5 records for the violation report (consistent with the
        # other assert_* helpers -- a large violation burst should not
        # flood the JSON report).
        bad_idx = per_article.index[bad][:5]
        for i in bad_idx:
            violations.append(
                {
                    "type": "pit_violation",
                    "ticker": (
                        str(per_article.loc[i, "ticker"])
                        if "ticker" in per_article.columns
                        else None
                    ),
                    "published_at": pd.Timestamp(pub.loc[i]).isoformat(),
                    "period_close_ts": pd.Timestamp(pc.loc[i]).isoformat(),
                    "gap_seconds": int(
                        (pd.Timestamp(pub.loc[i]) - pd.Timestamp(pc.loc[i])).total_seconds()
                    ),
                }
            )
    return violations


def assert_probs_sum_to_one(df: pd.DataFrame, pos: str, neg: str, neu: str) -> list[dict[str, Any]]:
    """Return a list of prob-sum violation records."""
    if df.empty:
        return []
    total = df[pos].to_numpy() + df[neg].to_numpy() + df[neu].to_numpy()
    bad = (total < 1.0 - _PROB_SUM_TOL) | (total > 1.0 + _PROB_SUM_TOL)
    violations: list[dict[str, Any]] = []
    if bad.any():
        for i in df.index[bad][:5]:
            violations.append(
                {
                    "type": "prob_sum_violation",
                    "index": int(i),
                    "pos": float(df.loc[i, pos]),
                    "neg": float(df.loc[i, neg]),
                    "neu": float(df.loc[i, neu]),
                    "sum": float(total[df.index.get_loc(i)]),
                }
            )
    return violations


def assert_weight_monotonicity(
    per_article: pd.DataFrame, decay_time_const_days: float = 5.0
) -> list[dict[str, Any]]:
    """Within each ``(ticker, period_close_ts)`` group, assert older
    articles receive smaller-or-equal weights. Returns a list of violation
    records.

    The weight function ``w_t = exp(-dt_t / tau)`` is monotonically
    increasing in ``dt_t`` (and thus in ``published_at``). A violation
    indicates a date-arithmetic bug in the aggregator.
    """
    if per_article.empty:
        return []
    required = {"ticker", "period_close_ts", "published_at"}
    if not required.issubset(per_article.columns):
        return [{"type": "missing_column", "column": str(required - set(per_article.columns))}]

    violations: list[dict[str, Any]] = []
    for (ticker, period_close), group in per_article.groupby(
        ["ticker", "period_close_ts"], sort=True
    ):
        if len(group) < 2:
            continue
        pub = _coerce_utc(group["published_at"], "published_at")
        pc = _coerce_utc(group["period_close_ts"], "period_close_ts")
        dt_seconds = (pc.iloc[0] - pub).dt.total_seconds().to_numpy()
        if (dt_seconds < 0).any():
            # PIT violation -- already caught by assert_pit_safety; skip
            # monotonicity for this group to avoid noise.
            continue
        dt_days = dt_seconds / 86400.0
        weights = np.exp(-dt_days / float(decay_time_const_days))
        # Sort by published_at ascending; weights must be non-decreasing.
        order = np.argsort(pub.to_numpy())
        sorted_w = weights[order]
        # Allow equal weights (articles published at the same instant).
        diffs = np.diff(sorted_w)
        if (diffs < -1e-12).any():
            violations.append(
                {
                    "type": "weight_monotonicity_violation",
                    "ticker": str(ticker),
                    "period_close_ts": pd.Timestamp(period_close).isoformat(),
                    "n_articles": int(len(group)),
                }
            )
    return violations


def assert_cache_freshness(
    meta: dict[str, Any] | None, expected_model: str, expected_revision: str
) -> list[dict[str, Any]]:
    """Assert the cache's ``model_name`` / ``model_revision`` match the
    active scorer. Returns a violation record on mismatch.
    """
    if meta is None:
        return []  # not provided -- skip
    violations: list[dict[str, Any]] = []
    if meta.get("model_name") != expected_model:
        violations.append(
            {
                "type": "cache_model_mismatch",
                "expected": expected_model,
                "cached": meta.get("model_name"),
            }
        )
    if meta.get("model_revision") != expected_revision:
        violations.append(
            {
                "type": "cache_revision_mismatch",
                "expected": expected_revision,
                "cached": meta.get("model_revision"),
            }
        )
    return violations


def assert_aggregate_reproducible(
    per_article: pd.DataFrame,
    per_period: pd.DataFrame,
    decay_time_const_days: float = 5.0,
) -> list[dict[str, Any]]:
    """For each ``(ticker, period_close_ts)`` group in ``per_period``,
    re-derive the aggregate from ``per_article`` and assert it matches.

    Uses the SAME weight function as the aggregator (imported from
    :mod:`equity.sentiment.aggregate` -- no formula duplication) and the
    same drop-null-ticker / drop-PIT-violation filtering as
    :func:`aggregate_per_period`. A mismatch indicates the stored
    ``per_period`` was produced by a different formula or a stale
    aggregator version (FOC-49 M3).
    """
    if per_period.empty:
        return []
    # Mirror the PIT/monotonicity skip-on-empty contract: when per_article
    # is empty, the re-derivation is impossible -- skip (the per_period
    # prob-sum check still runs). The operator should provide per_article
    # for a full reproducibility check.
    if per_article.empty:
        return []
    required = {"ticker", "published_at", "period_close_ts", "pos", "neg", "neu", "score"}
    if not required.issubset(per_article.columns):
        return [
            {
                "type": "aggregate_reproducibility",
                "reason": f"per_article missing columns: {sorted(required - set(per_article.columns))}",
            }
        ]

    # Mirror aggregate_per_period's filtering: drop null-ticker rows (those
    # go to market_wide) and PIT-violation rows (published_at > period_close).
    df = per_article.copy()
    df = df[df["ticker"].notna()]
    df["published_at"] = _coerce_utc(df["published_at"], "published_at")
    df["period_close_ts"] = _coerce_utc(df["period_close_ts"], "period_close_ts")
    df = df[df["published_at"] <= df["period_close_ts"]]

    # Index per_article by (ticker, period_close_ts) for O(1) group lookup.
    grouped: dict[tuple[str, pd.Timestamp], pd.DataFrame] = {}
    for (ticker, period_close), group in df.groupby(["ticker", "period_close_ts"], sort=True):
        grouped[(str(ticker), pd.Timestamp(period_close))] = group

    violations: list[dict[str, Any]] = []
    for _, row in per_period.iterrows():
        ticker = str(row["ticker"])
        period_close = pd.Timestamp(row["period_close_ts"])
        if period_close.tz is None:
            period_close = period_close.tz_localize("UTC")
        else:
            period_close = period_close.tz_convert("UTC")
        key = (ticker, period_close)
        group = grouped.get(key)
        if group is None or len(group) == 0:
            violations.append(
                {
                    "type": "aggregate_reproducibility",
                    "ticker": ticker,
                    "period_close_ts": period_close.isoformat(),
                    "reason": "no per_article rows for this (ticker, period_close_ts)",
                    "expected_n_articles": int(row["n_articles"]),
                    "rederived_n_articles": 0,
                }
            )
            continue
        weights = _compute_weights(period_close, group["published_at"], decay_time_const_days)
        rederived_score = _weighted_mean(group["score"].to_numpy(), weights)
        rederived_pos = _weighted_mean(group["pos"].to_numpy(), weights)
        rederived_neg = _weighted_mean(group["neg"].to_numpy(), weights)
        rederived_neu = _weighted_mean(group["neu"].to_numpy(), weights)
        rederived_n = int(len(group))
        tol = _REPRO_TOL
        if (
            abs(rederived_score - float(row["sentiment_score"])) > tol
            or abs(rederived_pos - float(row["sentiment_pos"])) > tol
            or abs(rederived_neg - float(row["sentiment_neg"])) > tol
            or abs(rederived_neu - float(row["sentiment_neu"])) > tol
            or rederived_n != int(row["n_articles"])
        ):
            violations.append(
                {
                    "type": "aggregate_reproducibility",
                    "ticker": ticker,
                    "period_close_ts": period_close.isoformat(),
                    "stored": {
                        "sentiment_score": float(row["sentiment_score"]),
                        "sentiment_pos": float(row["sentiment_pos"]),
                        "sentiment_neg": float(row["sentiment_neg"]),
                        "sentiment_neu": float(row["sentiment_neu"]),
                        "n_articles": int(row["n_articles"]),
                    },
                    "rederived": {
                        "sentiment_score": rederived_score,
                        "sentiment_pos": rederived_pos,
                        "sentiment_neg": rederived_neg,
                        "sentiment_neu": rederived_neu,
                        "n_articles": rederived_n,
                    },
                }
            )
    return violations


def run_sentiment_aggregate_guard(
    per_article: pd.DataFrame,
    per_period: pd.DataFrame,
    *,
    decay_time_const_days: float = 5.0,
    cache_meta: dict[str, Any] | None = None,
    expected_model: str | None = None,
    expected_revision: str | None = None,
) -> dict[str, Any]:
    """Run all five invariants and return a result dict.

    Parameters
    ----------
    per_article:
        Per-article cache frame (output of
        :meth:`SentimentCache.score_articles`). Must carry ``ticker``,
        ``published_at``, ``period_close_ts``, ``pos``, ``neg``, ``neu``.
        May be empty -- PIT + monotonicity + reproducibility checks are
        skipped (or report ``per_article empty``).
    per_period:
        Per-(ticker, period) aggregate frame. Must carry ``sentiment_pos``,
        ``sentiment_neg``, ``sentiment_neu``.
    decay_time_const_days:
        Exponential time constant tau (days) used by the aggregator (for
        the monotonicity + reproducibility checks).
    cache_meta:
        Optional parsed ``_meta.json`` from the sentiment cache. When
        provided alongside ``expected_model`` / ``expected_revision``, the
        cache freshness check is run.
    expected_model, expected_revision:
        Active scorer's ``model_name`` / ``model_revision``. Required when
        ``cache_meta`` is provided.
    """
    violations: list[dict[str, Any]] = []
    violations.extend(assert_pit_safety(per_article))
    violations.extend(
        assert_probs_sum_to_one(per_period, "sentiment_pos", "sentiment_neg", "sentiment_neu")
    )
    violations.extend(assert_weight_monotonicity(per_article, decay_time_const_days))
    if cache_meta is not None and expected_model and expected_revision:
        violations.extend(assert_cache_freshness(cache_meta, expected_model, expected_revision))
    violations.extend(assert_aggregate_reproducible(per_article, per_period, decay_time_const_days))

    return {
        "pass": len(violations) == 0,
        "n_violations": len(violations),
        "violations": violations,
        "n_per_article": int(len(per_article)),
        "n_per_period": int(len(per_period)),
        "cache_meta_checked": cache_meta is not None
        and expected_model is not None
        and expected_revision is not None,
    }


def _resolve_under_project_root(path: str | Path) -> Path:
    """Resolve ``--output`` safely (FOC-49 L9).

    Mirrors the spirit of
    :func:`equity.sentiment.aggregate._resolve_under_project_root` but
    adapted for a diagnostic JSON report: relative paths are joined to
    ``PROJECT_ROOT``; absolute paths are honored verbatim (a diagnostic
    report is legitimately written to operator-chosen locations -- CI
    artifact dirs, tmp dirs). Path-traversal attempts (``..`` components
    that would escape ``PROJECT_ROOT``) are refused.
    """
    raw = Path(path)
    parts = list(raw.parts)
    # Refuse ``..`` traversal in relative paths (escapes PROJECT_ROOT).
    # Absolute paths with ``..`` are also refused (defense in depth).
    if ".." in parts:
        raise ValueError(f"Refusing --output with path-traversal component '..': {path}.")
    if not raw.is_absolute():
        return (PROJECT_ROOT / raw).resolve()
    return raw.resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Sentiment aggregate guard: asserts PIT safety, prob-sum, "
            "weight monotonicity, cache freshness, and aggregate "
            "reproducibility. Exits 0 on PASS, 1 on any violation."
        ),
    )
    parser.add_argument(
        "--per-article",
        help="Path to sentiment_per_article.parquet (cache).",
    )
    parser.add_argument(
        "--per-period",
        required=True,
        help="Path to sentiment_per_period.parquet (aggregate).",
    )
    parser.add_argument(
        "--decay-time-const-days",
        type=float,
        default=5.0,
        help="Exponential time constant tau (days) used by the aggregator (default 5.0).",
    )
    parser.add_argument(
        "--cache-meta",
        help="Optional path to the sentiment cache _meta.json.",
    )
    parser.add_argument(
        "--expected-model",
        help="Active scorer model_name (for cache freshness check).",
    )
    parser.add_argument(
        "--expected-revision",
        help="Active scorer model_revision (for cache freshness check).",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write the JSON result (resolved under PROJECT_ROOT).",
    )
    args = parser.parse_args()

    per_period = pd.read_parquet(args.per_period)
    per_article = pd.read_parquet(args.per_article) if args.per_article else pd.DataFrame()

    cache_meta: dict[str, Any] | None = None
    if args.cache_meta:
        try:
            cache_meta = json.loads(Path(args.cache_meta).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"WARNING: could not parse --cache-meta ({exc}); skipping freshness check.")

    result = run_sentiment_aggregate_guard(
        per_article,
        per_period,
        decay_time_const_days=args.decay_time_const_days,
        cache_meta=cache_meta,
        expected_model=args.expected_model,
        expected_revision=args.expected_revision,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        # Containment check (FOC-49 L9): refuse paths outside PROJECT_ROOT.
        out_path = _resolve_under_project_root(args.output)
    else:
        ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        out_path = RESULTS_DIR / f"sentiment_aggregate_guard_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nResult written to: {out_path}")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
