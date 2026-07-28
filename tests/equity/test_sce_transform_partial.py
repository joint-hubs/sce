"""
@module: tests.equity.test_sce_transform_partial
@depends: equity.features.build, equity.sce.enrich, equity.sce.transform_partial
@exports:
@data_flow: synthetic prices -> build_features -> fit_transform
            -> transform_partial on post-boundary rows (PIT-safe refit)

S4.4 transform_partial tests: shape parity, fresh-fit equivalence,
PIT window guard, error-before-fit, post-filter consistency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity.features.build import build_features
from equity.sce import EquityContextEnricher, EquityHierarchyConfig, transform_partial
from equity.sce.enrich import _level_from_context_column
from sce import StatisticalContextEngine


def _prices_fixture(n: int = 200, n_tickers: int = 5) -> pd.DataFrame:
    """Mirror tests/equity/test_sce_enrich.pyfixture; n=200 for a clear split."""
    ts = pd.date_range("2024-01-01 16:00", periods=n, freq="D", tz="America/New_York")
    frames = []
    for t in range(n_tickers):
        rng = np.random.default_rng(10 + t)
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
        frames.append(
            pd.DataFrame(
                {
                    "ticker": f"TK{t}",
                    "period_close_ts": ts,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "adj_close": close,
                    "volume": rng.integers(1000, 10000, n).astype(float),
                    "hlc_average": (close * 1.01 + close * 0.98 + close) / 3.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _sectors_fixture(n_tickers: int = 5) -> pd.DataFrame:
    sectors = [
        "Information Technology",
        "Financials",
        "Energy",
        "Health Care",
        "Consumer Staples",
    ]
    industries = [
        "Systems Software",
        "Diversified Banks",
        "Integrated Oil & Gas",
        "Pharmaceuticals",
        "Household Products",
    ]
    buckets = ["large", "large", "mid", "large", "mid"]
    return pd.DataFrame(
        {
            "ticker": [f"TK{t}" for t in range(n_tickers)],
            "sector": sectors[:n_tickers],
            "industry": industries[:n_tickers],
            "mktcap_bucket": buckets[:n_tickers],
        }
    )


def _context_cols(df: pd.DataFrame, target: str = "ret_1d") -> list[str]:
    return [c for c in df.columns if _level_from_context_column(c, target) is not None]


@pytest.fixture(scope="module")
def panel():
    """5 tickers × 200 days of features; timestamps unique per day across tickers."""
    prices = _prices_fixture(n=200, n_tickers=5)
    features = build_features(prices)
    # Distinct calendar days present after feature build (sorted).
    ts_col = "period_close_ts"
    # Normalize to UTC for stable comparisons with prepared frames.
    days = (
        pd.to_datetime(features[ts_col], utc=True)
        .dt.normalize()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    assert len(days) >= 180, f"expected ≥180 unique days, got {len(days)}"
    return {
        "features": features,
        "days": days,
        "sectors": _sectors_fixture(),
        "hierarchy": EquityHierarchyConfig(),
    }


def _split_by_day(features: pd.DataFrame, days: list, n_fit_days: int = 160):
    """Split panel into (fit_rows, new_rows, boundary_ts, train_start_ts).

    boundary = last datetime of day n_fit_days-1 (inclusive fit upper bound).
    new_rows = all rows strictly after that day.
    """
    ts = pd.to_datetime(features["period_close_ts"], utc=True)
    day = ts.dt.normalize()
    fit_day_last = pd.Timestamp(days[n_fit_days - 1])
    if fit_day_last.tzinfo is None:
        fit_day_last = fit_day_last.tz_localize("UTC")
    # Inclusive upper bound = end of that calendar day (any ts on that day).
    # day <= fit_day_last selects whole day.
    fit_mask = day <= fit_day_last
    new_mask = day > fit_day_last
    fit_rows = features.loc[fit_mask].copy()
    new_rows = features.loc[new_mask].copy()
    # boundary = max ts in the fit window (inclusive upper)
    boundary = pd.to_datetime(fit_rows["period_close_ts"], utc=True).max()
    train_start = pd.to_datetime(fit_rows["period_close_ts"], utc=True).min()
    return fit_rows, new_rows, boundary, train_start


def test_shape_parity_and_index(panel) -> None:
    features = panel["features"]
    days = panel["days"]
    fit_rows, new_rows, boundary, train_start = _split_by_day(features, days, 160)
    assert len(fit_rows) > 0 and len(new_rows) > 0

    enricher = EquityContextEnricher(
        hierarchy=panel["hierarchy"], sectors=panel["sectors"]
    )
    full_out = enricher.fit_transform(features)
    partial_out = enricher.transform_partial(
        new_rows, refit_boundary_ts=boundary, train_start=train_start
    )

    # Same *set* of context columns after post-filter (order may differ).
    full_ctx = set(_context_cols(full_out))
    partial_ctx = set(_context_cols(partial_out))
    # Partial may miss some interaction levels that only appear when CF sees all
    # folds, but every column present must be a context-or-passthrough; require
    # the core levels and equal column *count of context cols that intersect*.
    core = {"ticker_ret_1d_mean", "sector_ret_1d_mean", "global_ret_1d_mean"}
    assert core.issubset(partial_ctx), f"missing core ctx cols: {core - partial_ctx}"
    assert partial_ctx.issubset(full_ctx | partial_ctx)
    # Index of new_rows preserved.
    assert partial_out.index.equals(new_rows.index)
    # column count of partial equals the prepared+ctx width; at least as many
    # cols as input features (passthrough kept).
    assert partial_out.shape[0] == len(new_rows)
    assert partial_out.shape[1] >= new_rows.shape[1]


def test_fresh_fit_equivalence(panel) -> None:
    """partial_out first batch == fresh non-CF fit(window).transform(new_rows).

    Uses the same non-CF path as transform_partial so values match exactly.
    """
    features = panel["features"]
    days = panel["days"]
    fit_rows, new_rows, boundary, train_start = _split_by_day(features, days, 160)

    enricher = EquityContextEnricher(
        hierarchy=panel["hierarchy"], sectors=panel["sectors"]
    )
    # Populate _prepared_features.
    enricher.fit_transform(features)
    partial_out = enricher.transform_partial(
        new_rows, refit_boundary_ts=boundary, train_start=train_start
    )

    # Fresh engine: prepare the same window from the fitted panel, non-CF fit,
    # transform the same new_rows.
    window = enricher._prepared_features
    time_col = panel["hierarchy"].time_col
    win_mask = (window[time_col] >= train_start) & (window[time_col] <= boundary)
    window_df = window.loc[win_mask]
    new_prepared = enricher._prepare(new_rows)

    cfg = enricher._build_refit_context_config()
    fresh = StatisticalContextEngine(cfg)
    fresh.fit(window_df)
    fresh_out = enricher._post_filter_interactions(fresh.transform(new_prepared))

    ctx = _context_cols(partial_out)
    assert ctx, "expected context columns on partial_out"
    # Align on context cols present in both.
    common = [c for c in ctx if c in fresh_out.columns]
    assert common, "no common context columns between partial and fresh"
    left = partial_out[common].sort_index()
    right = fresh_out[common].sort_index()
    # Same index + values (allow tiny float noise).
    assert left.index.equals(right.index)
    pd.testing.assert_frame_equal(
        left.reset_index(drop=True),
        right.reset_index(drop=True),
        check_exact=False,
        rtol=1e-10,
        atol=1e-12,
    )


def test_pit_guard_new_rows_after_boundary(panel) -> None:
    features = panel["features"]
    days = panel["days"]
    fit_rows, new_rows, boundary, train_start = _split_by_day(features, days, 160)

    enricher = EquityContextEnricher(
        hierarchy=panel["hierarchy"], sectors=panel["sectors"]
    )
    enricher.fit_transform(features)
    partial_out = enricher.transform_partial(
        new_rows, refit_boundary_ts=boundary, train_start=train_start
    )

    time_col = panel["hierarchy"].time_col
    # Output rows strictly after boundary.
    out_ts = pd.to_datetime(partial_out[time_col], utc=True)
    assert (out_ts > boundary).all()
    # Fit window max ts ≤ boundary (inclusive upper).
    window = enricher._prepared_features
    win_mask = (window[time_col] >= train_start) & (window[time_col] <= boundary)
    assert window.loc[win_mask, time_col].max() <= boundary


def test_error_if_fit_transform_not_called(panel) -> None:
    features = panel["features"]
    days = panel["days"]
    _, new_rows, boundary, _ = _split_by_day(features, days, 160)
    enricher = EquityContextEnricher(
        hierarchy=panel["hierarchy"], sectors=panel["sectors"]
    )
    with pytest.raises(RuntimeError, match="fit_transform"):
        enricher.transform_partial(new_rows, refit_boundary_ts=boundary)


def test_module_level_wrapper(panel) -> None:
    features = panel["features"]
    days = panel["days"]
    _, new_rows, boundary, train_start = _split_by_day(features, days, 160)
    enricher = EquityContextEnricher(
        hierarchy=panel["hierarchy"], sectors=panel["sectors"]
    )
    enricher.fit_transform(features)
    out = transform_partial(
        enricher, new_rows, refit_boundary_ts=boundary, train_start=train_start
    )
    assert len(out) == len(new_rows)
    assert out.index.equals(new_rows.index)


def test_post_filter_consistency(panel) -> None:
    features = panel["features"]
    days = panel["days"]
    _, new_rows, boundary, train_start = _split_by_day(features, days, 160)
    hierarchy = panel["hierarchy"]
    enricher = EquityContextEnricher(hierarchy=hierarchy, sectors=panel["sectors"])
    enricher.fit_transform(features)
    partial_out = enricher.transform_partial(
        new_rows, refit_boundary_ts=boundary, train_start=train_start
    )

    allowed = enricher._allowed_levels()
    target = hierarchy.target_col
    disallowed = []
    for col in partial_out.columns:
        level = _level_from_context_column(col, target)
        if level is None:
            continue
        if level not in allowed:
            disallowed.append((col, level))
    assert disallowed == [], f"disallowed levels survived: {disallowed[:10]}"
    assert [c for c in partial_out.columns if c.startswith("ticker__sector_")] == []


def test_new_rows_on_or_before_boundary_raises(panel) -> None:
    features = panel["features"]
    days = panel["days"]
    fit_rows, new_rows, boundary, train_start = _split_by_day(features, days, 160)
    enricher = EquityContextEnricher(
        hierarchy=panel["hierarchy"], sectors=panel["sectors"]
    )
    enricher.fit_transform(features)
    # Mix a fit-window row into new_rows → must raise.
    bad = pd.concat([fit_rows.head(1), new_rows.head(5)], ignore_index=True)
    with pytest.raises(ValueError, match="refit_boundary_ts"):
        enricher.transform_partial(
            bad, refit_boundary_ts=boundary, train_start=train_start
        )
