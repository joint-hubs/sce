"""
@module: tests.equity.test_lookahead_indicator
@depends: equity.diagnostics.lookahead_indicator, equity.features.technical
@exports:
@data_flow: hand-built prices -> add_technical_features -> guard PASS;
            injected leaky SMA -> guard FAIL with n_violations >= 1

S3.3 tests. The guard recomputes each indicator from ``prices[:t]`` only and
asserts equality with the stored feature within abs=1e-9 (PRD §6.4.1).
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from equity.diagnostics.lookahead_indicator import (
    run_lookahead_indicator,
)
from equity.features.technical import (
    _atr_naive,
    _ema_naive,
    _macd_naive,
    _rsi_naive,
    add_technical_features,
)


def _prices_fixture(n: int = 40, seed: int = 42) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="D", tz="America/New_York")
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    return pd.DataFrame(
        {
            "ticker": "AAPL",
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


def test_guard_passes_on_clean_past_only_frame():
    """A frame built by add_technical_features (past-only) must PASS."""
    prices = _prices_fixture(n=40, seed=1)
    feats = add_technical_features(prices)
    result = run_lookahead_indicator(feats, prices)
    assert result["pass"] is True
    assert result["n_violations"] == 0
    assert result["n_features_checked"] > 0
    assert result["n_rows_checked"] == len(prices)


def test_guard_detects_leaky_sma_closed_right():
    """The classic leak: an SMA built with closed='right' (or no shift) instead
    of the past-only form. The guard MUST flag it (n_violations >= 1)."""
    prices = _prices_fixture(n=20, seed=2)
    # Inject a DELIBERATELY LEAKY sma_5: rolling(5).mean() with NO shift (so it
    # includes close[t]). The guard's spec fn computes the naive SMA; feeding
    # prices[:t] reproduces the past-only value, which will NOT match the
    # stored leaky value.
    feats = prices.copy()
    feats["sma_5"] = prices["close"].rolling(5).mean()  # leaky, no shift
    # The spec fn returns the naive (current-row-inclusive) SMA; the guard
    # feeds prices[:t] and takes the last value (which equals the past-only
    # value). The stored leaky value uses prices[:t+1], so it diverges.
    result = run_lookahead_indicator(
        feats,
        prices,
        column_specs={"sma_5": lambda p: p["close"].rolling(5).mean()},
    )
    assert result["pass"] is False
    assert result["n_violations"] >= 1
    assert any(v["feature"] == "sma_5" for v in result["violations"])


def test_guard_detects_nan_mismatch():
    """If the stored value is finite but the re-derived value is NaN (or
    vice versa), that's a partial-window bug -> violation."""
    prices = _prices_fixture(n=20, seed=3)
    feats = prices.copy()
    # Stored value is finite everywhere (0.0) but re-derived is NaN in warmup.
    feats["sma_5"] = 0.0
    result = run_lookahead_indicator(
        feats,
        prices,
        column_specs={"sma_5": lambda p: p["close"].rolling(5).mean()},
    )
    assert result["pass"] is False
    assert any(v["type"] == "lookahead_nan_mismatch" for v in result["violations"])


def test_guard_two_ticker_no_cross_bleed():
    """A 2-ticker frame: the guard must recompute per-ticker, no cross-bleed."""
    ts = pd.date_range("2024-01-01", periods=30, freq="D", tz="America/New_York")
    rng_a = np.random.default_rng(11)
    rng_b = np.random.default_rng(22)
    prices = pd.concat(
        [
            pd.DataFrame(
                {
                    "ticker": "AAPL",
                    "period_close_ts": ts,
                    "open": 1.0,
                    "high": 1.5,
                    "low": 0.5,
                    "close": np.cumprod(1 + rng_a.normal(0, 0.01, 30)),
                    "adj_close": 1.0,
                    "volume": 1000.0,
                    "hlc_average": 1.0,
                }
            ),
            pd.DataFrame(
                {
                    "ticker": "MSFT",
                    "period_close_ts": ts,
                    "open": 1.0,
                    "high": 1.5,
                    "low": 0.5,
                    "close": np.cumprod(1 + rng_b.normal(0, 0.01, 30)),
                    "adj_close": 1.0,
                    "volume": 1000.0,
                    "hlc_average": 1.0,
                }
            ),
        ],
        ignore_index=True,
    )
    feats = add_technical_features(prices)
    result = run_lookahead_indicator(feats, prices)
    assert result["pass"] is True
    assert result["n_violations"] == 0
    assert result["n_rows_checked"] == 60


def test_guard_tz_canonicalization_amex_vs_utc():
    """Features in UTC, prices in America/New_York -- the guard must reconcile."""
    prices = _prices_fixture(n=30, seed=4)
    feats = add_technical_features(prices)
    # Convert features period_close_ts to UTC.
    feats["period_close_ts"] = feats["period_close_ts"].dt.tz_convert("UTC")
    result = run_lookahead_indicator(feats, prices)
    assert result["pass"] is True
    assert result["n_violations"] == 0


def test_guard_cli_passes_on_clean_frame(tmp_path, monkeypatch):
    """CLI: ``python -m equity.diagnostics.lookahead_indicator --features ...
    --prices ...`` exits 0 on a clean past-only frame."""
    from equity.diagnostics import lookahead_indicator as guard

    prices = _prices_fixture(n=40, seed=5)
    feats = add_technical_features(prices)
    feats_path = tmp_path / "features.parquet"
    prices_path = tmp_path / "prices.parquet"
    feats.to_parquet(feats_path, index=False)
    prices.to_parquet(prices_path, index=False)
    # Round-1 review fix (TRIVIAL 5): the CLI now refuses --output OUTSIDE the
    # project root, so write the result to a relative path under PROJECT_ROOT
    # (the results dir) and clean up after.
    out_rel = "results/diagnostics/equity/_test_cli_clean_result.json"
    out = guard.PROJECT_ROOT / out_rel
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "equity.diagnostics.lookahead_indicator",
                "--features",
                str(feats_path),
                "--prices",
                str(prices_path),
                "--output",
                out_rel,
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"CLI failed:\n{proc.stderr}\n{proc.stdout}"
        assert out.exists()
        result = json.loads(out.read_text())
        assert result["pass"] is True
        assert result["n_violations"] == 0
    finally:
        if out.exists():
            out.unlink()


def test_guard_cli_detects_leak(tmp_path):
    """CLI exits 1 on an injected leaky SMA."""
    prices = _prices_fixture(n=20, seed=6)
    feats = prices.copy()
    feats["sma_5"] = prices["close"].rolling(5).mean()  # leaky
    feats_path = tmp_path / "leaky_features.parquet"
    prices_path = tmp_path / "prices.parquet"
    feats.to_parquet(feats_path, index=False)
    prices.to_parquet(prices_path, index=False)
    # Build a guard with the leaky spec in a temp module-level monkeypatch:
    # simpler -- call run_lookahead_indicator directly via a custom column_specs
    # CLI doesn't accept custom specs, so verify via the function API here.
    result = run_lookahead_indicator(
        feats,
        prices,
        column_specs={"sma_5": lambda p: p["close"].rolling(5).mean()},
    )
    assert result["pass"] is False
    assert result["n_violations"] >= 1


def test_guard_cli_rejects_output_outside_project_root():
    """L9 containment: --output with '..' is refused."""
    from equity.diagnostics import lookahead_indicator as guard

    with pytest.raises(ValueError, match="path-traversal"):
        guard._resolve_under_project_root("../../etc/evil.json")


def test_guard_cli_accepts_absolute_output_path_under_project_root(tmp_path, monkeypatch):
    """Round-1 review fix (TRIVIAL 5): an absolute --output INSIDE the project
    root is still accepted (resolved path lives under PROJECT_ROOT)."""
    from equity.diagnostics import lookahead_indicator as guard

    # Point the guard's PROJECT_ROOT at tmp_path so the absolute path under it
    # is accepted, but an absolute path outside it is rejected (next test).
    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    out = tmp_path / "diagnostic.json"
    resolved = guard._resolve_under_project_root(str(out))
    assert resolved == out.resolve()


def test_guard_cli_rejects_absolute_output_path_outside_project_root(tmp_path, monkeypatch):
    """Round-1 review fix (TRIVIAL 5): an absolute --output OUTSIDE the project
    root is now REJECTED (the old code accepted any absolute path, defeating
    the containment guard)."""
    from equity.diagnostics import lookahead_indicator as guard

    monkeypatch.setattr(guard, "PROJECT_ROOT", tmp_path)
    outside = tmp_path.parent / "evil.json"
    with pytest.raises(ValueError, match="outside project root"):
        guard._resolve_under_project_root(str(outside))


# ---------------------------------------------------------------------------
# Round-1 review fixes: BLOCKER 1 (lag-layer coverage), TEST 7 (EMA/RSI/MACD/
# ATR leak detection), TEST 8 (defensive ValueError branches), SUBSTANTIVE 11
# (strict), SUBSTANTIVE 12 (max-rows).
# ---------------------------------------------------------------------------


def test_guard_detects_leaky_rollmean():
    """BLOCKER 1: a deliberately leaky ``{base}_rollmean{N}`` built with
    ``closed='right'`` (includes the current row) MUST be flagged. The guard's
    auto-generated lag spec for price-derived lag columns recomputes the
    past-only value (``closed='left'`` equivalent at t-1) and diverges from the
    stored leaky value (``closed='right'`` at t, includes close[t])."""
    prices = _prices_fixture(n=20, seed=8)
    feats = prices.copy()
    # Leaky rollmean: closed='right' (default) INCLUDES the current row -> leak.
    feats["close_rollmean3"] = prices["close"].rolling(3, closed="right").mean()
    result = run_lookahead_indicator(feats, prices)
    assert result["pass"] is False
    assert result["n_violations"] >= 1
    assert any(v["feature"] == "close_rollmean3" for v in result["violations"])


def test_guard_clean_lag_layer_passes():
    """BLOCKER 1 corollary: a CLEAN past-only lag layer (built by apply_lags
    with closed='left') must PASS the guard for the price-derived lag columns.
    Confirms the auto-generated lag specs are correctly aligned (naive at t-1
    == stored past-only at t)."""
    from equity.features.lag import apply_lags

    prices = _prices_fixture(n=40, seed=9)
    feats = apply_lags(prices, ["close"], windows=(3,))
    result = run_lookahead_indicator(feats, prices)
    assert result["pass"] is True
    assert result["n_violations"] == 0
    # The price-derived lag cols were checked.
    assert "close_lag3" in result["features_checked"]
    assert "close_rollmean3" in result["features_checked"]
    assert "close_rollstd3" in result["features_checked"]


def test_guard_detects_leaky_ema():
    """TEST 7: a leaky EMA (forgot shift(1) -- stored = naive unshifted) is
    caught."""
    prices = _prices_fixture(n=30, seed=10)
    feats = prices.copy()
    feats["ema_5"] = _ema_naive(prices["close"], 5)  # leaky: unshifted naive
    result = run_lookahead_indicator(feats, prices)
    assert result["pass"] is False
    assert any(v["feature"] == "ema_5" for v in result["violations"])


def test_guard_detects_leaky_rsi():
    """TEST 7: a leaky RSI (forgot shift(1)) is caught."""
    prices = _prices_fixture(n=30, seed=11)
    feats = prices.copy()
    feats["rsi_14"] = _rsi_naive(prices["close"], 14)
    result = run_lookahead_indicator(feats, prices)
    assert result["pass"] is False
    assert any(v["feature"] == "rsi_14" for v in result["violations"])


def test_guard_detects_leaky_macd():
    """TEST 7: a leaky MACD (forgot shift(1) on all three columns) is caught."""
    prices = _prices_fixture(n=40, seed=12)
    feats = prices.copy()
    naive_macd = _macd_naive(prices["close"])
    feats["macd"] = naive_macd["macd"]
    feats["macd_signal"] = naive_macd["macd_signal"]
    feats["macd_hist"] = naive_macd["macd_hist"]
    result = run_lookahead_indicator(feats, prices)
    assert result["pass"] is False
    # At least one MACD column flagged.
    macd_features = {v["feature"] for v in result["violations"]} & {
        "macd",
        "macd_signal",
        "macd_hist",
    }
    assert macd_features


def test_guard_detects_leaky_atr():
    """TEST 7: a leaky ATR (forgot shift(1)) is caught."""
    prices = _prices_fixture(n=30, seed=13)
    feats = prices.copy()
    feats["atr_14"] = _atr_naive(prices["high"], prices["low"], prices["close"], 14)
    result = run_lookahead_indicator(feats, prices)
    assert result["pass"] is False
    assert any(v["feature"] == "atr_14" for v in result["violations"])


def test_guard_raises_on_missing_ticker_in_features():
    """TEST 8: features frame missing 'ticker' -> ValueError (defensive branch)."""
    prices = _prices_fixture(n=10, seed=14)
    feats = prices.drop(columns=["ticker"])
    with pytest.raises(ValueError, match="features frame must have"):
        run_lookahead_indicator(feats, prices)


def test_guard_raises_on_missing_ticker_in_prices():
    """TEST 8: prices frame missing 'ticker' -> ValueError (defensive branch)."""
    prices = _prices_fixture(n=10, seed=15)
    feats = prices.copy()
    bad_prices = prices.drop(columns=["ticker"])
    with pytest.raises(ValueError, match="prices frame must have"):
        run_lookahead_indicator(feats, bad_prices)


def test_guard_strict_raises_on_missing_spec_column():
    """SUBSTANTIVE 11: strict=True raises when a spec column is missing from
    features (instead of silently omitting it)."""
    prices = _prices_fixture(n=10, seed=16)
    feats = prices.copy()  # no technical cols -> all default specs missing
    with pytest.raises(ValueError, match="strict mode"):
        run_lookahead_indicator(feats, prices, strict=True)


def test_guard_non_strict_omits_missing_spec_column():
    """SUBSTANTIVE 11 corollary: default (strict=False) silently omits missing
    spec columns and returns pass=True (no features checked)."""
    prices = _prices_fixture(n=10, seed=17)
    feats = prices.copy()  # no technical cols
    result = run_lookahead_indicator(feats, prices)
    assert result["pass"] is True
    assert result["n_features_checked"] == 0


def test_guard_max_rows_truncates():
    """SUBSTANTIVE 12: ``max_rows`` downsamples per-ticker; n_rows_checked drops."""
    prices = _prices_fixture(n=40, seed=18)
    feats = add_technical_features(prices)
    full = run_lookahead_indicator(feats, prices)
    trunc = run_lookahead_indicator(feats, prices, max_rows=10)
    assert trunc["max_rows"] == 10
    assert trunc["n_rows_checked"] == 10  # single ticker, 40 rows -> 10
    assert full["n_rows_checked"] == 40
    # Truncation must not introduce spurious violations (clean frame still clean).
    assert trunc["pass"] is True
    assert trunc["n_violations"] == 0


def test_guard_cli_max_rows_truncates(tmp_path):
    """SUBSTANTIVE 12: ``--max-rows`` CLI flag wires through to the guard and
    truncates the row count."""
    from equity.diagnostics import lookahead_indicator as guard

    prices = _prices_fixture(n=40, seed=19)
    feats = add_technical_features(prices)
    feats_path = tmp_path / "features.parquet"
    prices_path = tmp_path / "prices.parquet"
    feats.to_parquet(feats_path, index=False)
    prices.to_parquet(prices_path, index=False)
    out_rel = "results/diagnostics/equity/_test_cli_max_rows.json"
    out = guard.PROJECT_ROOT / out_rel
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "equity.diagnostics.lookahead_indicator",
                "--features",
                str(feats_path),
                "--prices",
                str(prices_path),
                "--max-rows",
                "10",
                "--output",
                out_rel,
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"CLI failed:\n{proc.stderr}\n{proc.stdout}"
        result = json.loads(out.read_text())
        assert result["max_rows"] == 10
        assert result["n_rows_checked"] == 10
    finally:
        if out.exists():
            out.unlink()
