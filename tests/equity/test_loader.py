"""
@module: tests.equity.test_loader
@depends: equity.data.loader, equity.data.registry
@exports:
@data_flow: seed universe CSV -> EquityDataLoader -> universe tuples
"""

from __future__ import annotations

from datetime import date

import pytest

from equity.data.loader import EquityDataLoader
from equity.data.registry import get_universe_info, list_universes


def test_list_universes_includes_sp500():
    names = {info.name for info in list_universes()}
    assert "sp500" in names


def test_get_universe_info_resolves_seed_file():
    info = get_universe_info("sp500")
    assert info.name == "sp500"
    assert info.universe_file.name == "sp500_universe.csv"
    assert info.exists_locally is True


def test_universe_returns_nonempty_three_tuples():
    loader = EquityDataLoader("sp500", "2008-01-01", "2008-12-31")
    universe = loader.universe()
    assert len(universe) > 0
    for entry in universe:
        assert len(entry) == 3
        ticker, listed_at, delisted_at = entry
        assert isinstance(ticker, str)
        assert listed_at is None or isinstance(listed_at, date)
        assert delisted_at is None or isinstance(delisted_at, date)


# Delisted tickers from the seed CSV with verified delisted_at dates.
DELISTED_TICKERS = [
    "ENE", "LEH", "WCOM", "BSC", "WB", "MER", "WM", "CFC",
    "SHLD", "GM", "RSH", "JCP", "MON",
]


def test_universe_has_at_least_ten_delistees_with_nonnull_delisted_at():
    # Wide window so every delisted ticker is alive-for-some-part and included.
    loader = EquityDataLoader("sp500", "2000-01-01", "2026-12-31")
    universe = loader.universe()
    by_ticker = {t: (l, d) for (t, l, d) in universe}
    non_null_delisted = [
        t for t in DELISTED_TICKERS
        if t in by_ticker and by_ticker[t][1] is not None
    ]
    assert len(non_null_delisted) >= 10, (
        f"Expected >=10 delistees with non-null delisted_at, got "
        f"{len(non_null_delisted)}: {non_null_delisted}"
    )


def test_loader_raises_on_nonexistent_ticker_filter():
    with pytest.raises(ValueError, match="NONEXISTENT"):
        EquityDataLoader(
            "sp500", "2008-01-01", "2008-12-31", tickers=["NONEXISTENT"]
        )


def test_loader_raises_on_ticker_delisted_before_start():
    # ENE delisted 2002-01-15; window starts 2010-01-01 -> outside window.
    with pytest.raises(ValueError, match="ENE"):
        EquityDataLoader(
            "sp500", "2010-01-01", "2010-12-31", tickers=["ENE"]
        )


def test_ticker_delisted_during_window_is_included():
    # LEH delisted 2008-09-22, inside [2008-01-01, 2008-12-31] -> included.
    loader = EquityDataLoader("sp500", "2008-01-01", "2008-12-31")
    universe = loader.universe()
    tickers = {t for (t, _l, _d) in universe}
    assert "LEH" in tickers
    leh_entry = [entry for entry in universe if entry[0] == "LEH"][0]
    assert leh_entry[2] == date(2008, 9, 22)
