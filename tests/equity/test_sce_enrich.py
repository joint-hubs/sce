"""
@module: tests.equity.test_sce_enrich
@depends: equity.features.build, equity.sce.enrich
@exports:
@data_flow: synthetic prices -> build_features -> EquityContextEnricher
            -> enriched frame with allow-listed SCE context columns

S4.1/S4.2 enricher integration tests on a synthetic multi-ticker panel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity.features.build import build_features
from equity.sce import EquityContextEnricher, EquityHierarchyConfig
from equity.sce.enrich import _level_from_context_column


def _prices_fixture(n: int = 100, n_tickers: int = 5) -> pd.DataFrame:
    """Mirror tests/equity/test_features_build.py:_prices_fixture.

    16:00 ET canonical session close. ``n=100`` rows × 5 tickers = 500 rows —
    enough for rolling CF (n_folds=5) with min_group_size=20.
    """
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
    """Synthetic hierarchy — two sectors so sector-level stats have support."""
    sectors = ["Information Technology", "Financials", "Energy", "Health Care", "Consumer Staples"]
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


@pytest.fixture(scope="module")
def enriched_bundle():
    """Build features + run enricher once per module (SCE CF is the slow bit)."""
    prices = _prices_fixture(n=100, n_tickers=5)
    features = build_features(prices)
    # Keep more rows after NaN warm-up of indicators: enricher does not drop
    # NaN targets itself; SCE's aggregator dropna's per group. Use the full
    # feature frame (NaN ret_1d on row 0 of each ticker is fine).
    hierarchy = EquityHierarchyConfig()
    # Lower min_group_size slightly? No — contract is 20; 5 tickers × ~100
    # rows gives ticker groups of 100 and sector groups of 100, plenty.
    enricher = EquityContextEnricher(hierarchy=hierarchy, sectors=_sectors_fixture())
    out = enricher.fit_transform(features)
    return {
        "features": features,
        "out": out,
        "enricher": enricher,
        "hierarchy": hierarchy,
    }


def test_output_has_ret_1d_alias(enriched_bundle) -> None:
    out = enriched_bundle["out"]
    assert "ret_1d" in out.columns
    # Alias equals the past-only source column where both are non-null.
    both = out["ret_1d"].notna() & out["ret_1d_log"].notna()
    assert both.any()
    pd.testing.assert_series_equal(
        out.loc[both, "ret_1d"],
        out.loc[both, "ret_1d_log"],
        check_names=False,
    )


def test_context_columns_present(enriched_bundle) -> None:
    out = enriched_bundle["out"]
    # At least one of the canonical singleton / global mean columns.
    expected_any = {
        "ticker_ret_1d_mean",
        "sector_ret_1d_mean",
        "global_ret_1d_mean",
    }
    present = expected_any.intersection(out.columns)
    assert present, (
        f"none of {sorted(expected_any)} present; sample cols="
        f"{[c for c in out.columns if 'ret_1d' in c][:20]}"
    )


def test_index_preserved(enriched_bundle) -> None:
    features = enriched_bundle["features"]
    out = enriched_bundle["out"]
    assert out.index.equals(features.index)


def test_no_disallowed_interaction_columns(enriched_bundle) -> None:
    out = enriched_bundle["out"]
    target = enriched_bundle["hierarchy"].target_col
    allowed = enriched_bundle["enricher"]._allowed_levels()

    disallowed_found = []
    for col in out.columns:
        level = _level_from_context_column(col, target)
        if level is None:
            continue
        if "__" in level and level not in allowed:
            disallowed_found.append((col, level))

    assert disallowed_found == [], f"disallowed interaction cols survived: {disallowed_found[:10]}"
    # Explicit canary: ticker×sector pair must never appear.
    ticker_sector_cols = [c for c in out.columns if c.startswith("ticker__sector_")]
    assert ticker_sector_cols == []


def test_last_fold_timestamps_monotonic(enriched_bundle) -> None:
    folds = enriched_bundle["enricher"]._last_fold_timestamps
    assert folds, "expected non-empty _last_fold_timestamps after rolling CF"
    for i, fold in enumerate(folds):
        assert "train_max" in fold and "val_min" in fold, fold
        train_max = fold["train_max"]
        val_min = fold["val_min"]
        # Rolling/time CF: train window ends at or before val window starts.
        assert train_max <= val_min, (
            f"fold {i}: train_max={train_max} > val_min={val_min} (leak across folds)"
        )


def test_level_parser_contract() -> None:
    """Pin the SCE column naming so Phase 2/3 can reuse the parser."""
    t = "ret_1d"
    assert _level_from_context_column("ticker_ret_1d_mean", t) == "ticker"
    assert _level_from_context_column("global_ret_1d_mean", t) == "global"
    assert (
        _level_from_context_column("sector__time_bucket_ret_1d_mean", t)
        == "sector__time_bucket"
    )
    assert (
        _level_from_context_column("sector__time_bucket_ret_1d_mean_fold_std", t)
        == "sector__time_bucket"
    )
    # Passthrough / non-context columns.
    assert _level_from_context_column("ret_1d_log", t) is None
    assert _level_from_context_column("close", t) is None
    assert _level_from_context_column("sector", t) is None


def test_missing_target_raises() -> None:
    # Frame without ret_1d / ret_1d_log.
    df = pd.DataFrame(
        {
            "ticker": ["TK0", "TK0"],
            "period_close_ts": pd.date_range(
                "2024-01-01 16:00", periods=2, freq="D", tz="America/New_York"
            ),
            "close": [100.0, 101.0],
        }
    )
    enricher = EquityContextEnricher(sectors=_sectors_fixture(1))
    with pytest.raises(ValueError, match="ret_1d"):
        enricher.fit_transform(df)
