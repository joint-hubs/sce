# Test configuration for equity package tests.
import os
import sys
from pathlib import Path

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
