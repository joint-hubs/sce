# Test configuration for equity package tests.
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root first so tests resolve local `equity` package, not a site-packages install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Repo root (configs/kaggle.json lives here).
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _network_or_yfinance_available() -> bool:
    """Return True if the yfinance live integration test should run.

    Gate: ``SCE_EQUITY_LIVE_TEST=1`` env var AND yfinance importable. Default
    CI runs do not set the env var, so the test is skipped (no network).
    """
    if os.environ.get("SCE_EQUITY_LIVE_TEST") != "1":
        return False
    try:
        import yfinance  # noqa: F401
    except ImportError:
        return False
    return True


def _kaggle_available() -> bool:
    """Return True if the Kaggle historical integration test should run.

    Gate (all required): explicit opt-in via ``SCE_EQUITY_LIVE_TEST=1`` env
    var (so the default run hits no network), ``data.download`` importable
    (it is NOT a packaged module -- only ``sce*``/``equity*`` are packaged),
    AND Kaggle credentials configured (``KAGGLE_USERNAME``/``KAGGLE_KEY`` env
    vars OR ``configs/kaggle.json`` present).

    The env-var opt-in is shared with the yfinance gate so a default ``pytest``
    run never touches the network; set ``SCE_EQUITY_LIVE_TEST=1`` to enable
    both live integration tests.
    """
    if os.environ.get("SCE_EQUITY_LIVE_TEST") != "1":
        return False
    try:
        from data.download import download_kaggle_file, parse_source  # noqa: F401
    except ImportError:
        return False
    has_env_creds = bool(
        os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    )
    has_creds_file = (_REPO_ROOT / "configs" / "kaggle.json").exists()
    return has_env_creds or has_creds_file


# ---------------------------------------------------------------------------
# Shared fixtures for S4/S5 (FOC-51/FOC-52)
# ---------------------------------------------------------------------------


def _prices_fixture(n: int = 100, n_tickers: int = 5) -> pd.DataFrame:
    """Synthetic multi-ticker OHLCV panel (16:00 ET closes).

    Mirror ``tests/equity/test_sce_enrich.py:_prices_fixture``. ``n=100`` × 5
    tickers = 500 rows — enough for rolling CF (n_folds=5).
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
    """Synthetic hierarchy — two-plus sectors so sector-level stats have support."""
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
def prices_fixture() -> pd.DataFrame:
    """Module-scoped synthetic prices (5 tickers × 100 sessions)."""
    return _prices_fixture(n=100, n_tickers=5)


@pytest.fixture(scope="module")
def sectors_fixture() -> pd.DataFrame:
    return _sectors_fixture(5)


@pytest.fixture(scope="module")
def long_panel() -> pd.DataFrame:
    """Long synthetic prices panel for h=21/h=63 horizon smoke (5 × 220 days)."""
    return _prices_fixture(n=220, n_tickers=5)


@pytest.fixture(scope="module")
def enriched_features(prices_fixture: pd.DataFrame, sectors_fixture: pd.DataFrame):
    """Build features + run SCE enricher once per module (slow CF recycled).

    Returns ``{prices, features, enriched, sectors, hierarchy}``. Used by S5
    forecaster tests so each file does not rebuild SCE.
    """
    from equity.features.build import build_features
    from equity.sce import EquityContextEnricher, EquityHierarchyConfig

    features = build_features(prices_fixture)
    hierarchy = EquityHierarchyConfig()
    enricher = EquityContextEnricher(hierarchy=hierarchy, sectors=sectors_fixture)
    enriched = enricher.fit_transform(features)
    return {
        "prices": prices_fixture,
        "features": features,
        "enriched": enriched,
        "sectors": sectors_fixture,
        "hierarchy": hierarchy,
        "enricher": enricher,
    }
