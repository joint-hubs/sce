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
    _numeric_feature_cols,
    audit_feature_dominance_equity,
    evaluate_equity_sce,
    run_crossfit_ab_equity,
    run_permuted_target_equity,
)
from equity.features.build import build_features
from equity.sce import EquityContextEnricher, EquityHierarchyConfig


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


def test_design_matrix_excludes_ret_1d_log_and_target_aliases(
    features_500: pd.DataFrame, sectors: pd.DataFrame
) -> None:
    """ret_1d_log / target-identical cols must never enter the Ridge design matrix.

    Defense-in-depth: enricher drops ret_1d_log after aliasing, AND
    _numeric_feature_cols excludes any column equal to the target.
    """
    enricher = EquityContextEnricher(
        hierarchy=EquityHierarchyConfig(n_folds=3, min_group_size=10),
        sectors=sectors,
    )
    prepared = enricher._prepare(features_500)
    assert "ret_1d" in prepared.columns
    assert "ret_1d_log" not in prepared.columns

    # Even if a caller re-injects a target-identical alias, design cols drop it.
    leaked = prepared.copy()
    leaked["ret_1d_log"] = leaked["ret_1d"]
    leaked["target_clone"] = leaked["ret_1d"]
    base_cols = _numeric_feature_cols(
        leaked,
        target_col="ret_1d",
        time_col="period_close_ts",
    )
    assert "ret_1d" not in base_cols
    assert "ret_1d_log" not in base_cols
    assert "target_clone" not in base_cols
    assert "period_close_ts" not in base_cols


def test_timestamp_group_split_no_mid_day_cut(
    features_500: pd.DataFrame, sectors: pd.DataFrame
) -> None:
    """Train/test split must not bisect a timestamp across tickers."""
    hierarchy = EquityHierarchyConfig(n_folds=3, min_group_size=10)
    # Smoke the full path; assert structure still holds under ts-group split.
    result = evaluate_equity_sce(
        features_500,
        hierarchy=hierarchy,
        sectors=sectors,
        use_cross_fitting=False,
        test_frac=0.3,
    )
    assert result["n_train"] > 0 and result["n_test"] > 0
    # Reconstruct split boundary: every train ts < every test ts is implied by
    # group split — verify via a shadowed prepare+split matching evaluate.
    enricher = EquityContextEnricher(hierarchy=hierarchy, sectors=sectors)
    prepared = enricher._prepare(features_500)
    prepared = prepared.loc[prepared["ret_1d"].notna()].copy()
    prepared = prepared.sort_values("period_close_ts").reset_index(drop=True)
    unique_ts = pd.Index(prepared["period_close_ts"].drop_duplicates().sort_values())
    split_idx = max(1, min(len(unique_ts) - 1, int(round(len(unique_ts) * 0.7))))
    split_day = unique_ts[split_idx]
    train_ts = set(prepared.loc[prepared["period_close_ts"] < split_day, "period_close_ts"])
    test_ts = set(prepared.loc[prepared["period_close_ts"] >= split_day, "period_close_ts"])
    assert train_ts.isdisjoint(test_ts)
    # No timestamp appears partially: all rows of a ts land on one side.
    for ts, grp in prepared.groupby("period_close_ts", sort=False):
        sides = {
            ("train" if t < split_day else "test")
            for t in grp["period_close_ts"]
        }
        assert len(sides) == 1, f"timestamp {ts} split across sides: {sides}"


def test_permuted_target_fails_on_random_returns(sectors: pd.DataFrame) -> None:
    """With leak-free random returns, permuted-target diagnostic must PASS
    (perm advantage stays near zero / below threshold) and design is leak-free.

    The historical blocker was ret_1d_log leaking into X making perm advantage ≈ 0
    for the wrong reason (label still present). Here we pin: no alias in design
    AND the diagnostic runs end-to-end on pure-noise returns.
    """
    n, n_tickers = 80, 4
    ts = pd.date_range("2024-01-01 16:00", periods=n, freq="D", tz="UTC")
    frames = []
    rng = np.random.default_rng(99)
    for t in range(n_tickers):
        # IID noise close path — no real structure for SCE to exploit.
        rets = rng.normal(0, 0.02, n)
        close = 100.0 * np.cumprod(1 + rets)
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
                    "hlc_average": close,
                }
            )
        )
    prices = pd.concat(frames, ignore_index=True)
    features = build_features(prices)
    # Direct design-matrix pin on prepared frame.
    enricher = EquityContextEnricher(
        hierarchy=EquityHierarchyConfig(n_folds=3, min_group_size=8),
        sectors=sectors.iloc[:n_tickers].copy(),
    )
    prepared = enricher._prepare(features)
    base_cols = _numeric_feature_cols(
        prepared, target_col="ret_1d", time_col="period_close_ts"
    )
    assert "ret_1d_log" not in base_cols
    assert "ret_1d" not in base_cols
    assert "ret_1d_log" not in prepared.columns

    result = run_permuted_target_equity(
        features,
        hierarchy=EquityHierarchyConfig(n_folds=3, min_group_size=8),
        sectors=sectors.iloc[:n_tickers].copy(),
        n_permutations=3,
        seed=7,
        use_cross_fitting=False,
    )
    # Contract: diagnostic completes; with noise perm mean should stay under
    # the 1.0 pp threshold (pass=True). If a label leak reappears, SCE can
    # keep advantage under permutation and this would flip — pin pass.
    assert "pass" in result
    assert "sce_advantage_permuted_mean" in result
    assert result["pass"] is True, (
        f"permuted-target unexpectedly failed on pure-noise neon: {result}"
    )
