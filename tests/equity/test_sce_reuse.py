"""
@module: tests.equity.test_sce_reuse
@depends: equity.features.build, equity.diagnostics.sce_reuse
@exports:
@data_flow: synthetic prices -> build_features -> evaluate_equity_sce /
            run_crossfit_ab_equity -> metrics dict

S4.6 equity-local SCE reuse runner smoke tests. Assertions are structural
(keys / types / finiteness) — not model-performance metabolites — so the
suite stays robust across sklearn / SCE internals drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity.diagnostics.sce_reuse import (
    audit_feature_dominance_equity,
    evaluate_equity_sce,
    run_crossfit_ab_equity,
)
from equity.features.build import build_features
from equity.sce import EquityHierarchyConfig


def _prices_fixture(n: int = 100, n_tickers: int = 5) -> pd.DataFrame:
    """Mirror tests/equity/test_sce_enrich.py:_prices_fixture."""
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


@pytest.fixture(scope="module")
def features_500() -> pd.DataFrame:
    """~500-row synthetic feature panel (5 tickers × 100 sessions)."""
    return build_features(_prices_fixture(n=100, n_tickers=5))


@pytest.fixture(scope="module")
def sectors() -> pd.DataFrame:
    return _sectors_fixture(5)


def test_evaluate_equity_sce_keys(features_500: pd.DataFrame, sectors: pd.DataFrame) -> None:
    # Lower min_group_size slightly so CF is reliable on the small smoke panel,
    # and keep a modest n_folds. Hierarchy default (20 / 5) already works on
    # 5×100 but Ridge+SCE is the slow bit — one call is enough.
    hierarchy = EquityHierarchyConfig(n_folds=3, min_group_size=10)
    result = evaluate_equity_sce(
        features_500,
        hierarchy=hierarchy,
        sectors=sectors,
        use_cross_fitting=False,  # faster & deterministic for smoke
    )

    for key in ("baseline_rmse", "baseline_r2", "sce_rmse", "sce_r2"):
        assert key in result, f"missing key {key!r} in {result.keys()}"
        assert isinstance(result[key], float), f"{key} not float: {type(result[key])}"
        assert np.isfinite(result[key]), f"{key} not finite: {result[key]}"

    assert result["n_train"] > 0
    assert result["n_test"] > 0
    assert result["n_features_baseline"] >= 0
    assert result["n_features_sce"] >= 0
    # Structural only — do not hard-assert sce_rmse <= baseline_rmse (Ridge +
    # concentration of small panels can wander). Soft check: not wildly worse.
    # If both finite the comparison ran end-to-end; that is the smoke contract.


def test_run_crossfit_ab_equity_leakage_signal(
    features_500: pd.DataFrame, sectors: pd.DataFrame
) -> None:
    hierarchy = EquityHierarchyConfig(n_folds=3, min_group_size=10)
    result = run_crossfit_ab_equity(
        features_500,
        hierarchy=hierarchy,
        sectors=sectors,
    )
    assert "leakage_signal_pp" in result
    assert isinstance(result["leakage_signal_pp"], float)
    assert np.isfinite(result["leakage_signal_pp"])
    assert "rmse_cf" in result and "rmse_no_cf" in result
    assert np.isfinite(result["rmse_cf"])
    assert np.isfinite(result["rmse_no_cf"])


def test_audit_feature_dominance_equity(tmp_path: Path) -> None:
    csv = tmp_path / "importance.csv"
    csv.write_text(
        "feature,avg_importance\n"
        "ticker_ret_1d_mean,0.50\n"
        "sector_ret_1d_mean,0.30\n"
        "sma_5,0.10\n"
        "close_lag1,0.10\n",
        encoding="utf-8",
    )
    result = audit_feature_dominance_equity(csv, top_k=2, threshold_pct=70.0)
    assert "top_k_share_pct" in result
    assert "dominated" in result
    assert result["top_k_share_pct"] == pytest.approx(80.0)
    assert result["dominated"] is True
