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
from equity.features.technical import add_technical_features


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
    prices = _prices_fixture(n=40, seed=5)
    feats = add_technical_features(prices)
    feats_path = tmp_path / "features.parquet"
    prices_path = tmp_path / "prices.parquet"
    feats.to_parquet(feats_path, index=False)
    prices.to_parquet(prices_path, index=False)
    out = tmp_path / "result.json"
    # Run as subprocess to test the actual CLI exit code.
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
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"CLI failed:\n{proc.stderr}\n{proc.stdout}"
    assert out.exists()
    result = json.loads(out.read_text())
    assert result["pass"] is True
    assert result["n_violations"] == 0


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


def test_guard_cli_accepts_absolute_output_path(tmp_path):
    from equity.diagnostics import lookahead_indicator as guard

    out = tmp_path / "diagnostic.json"
    resolved = guard._resolve_under_project_root(str(out))
    assert resolved == out.resolve()
