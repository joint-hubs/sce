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
    by_ticker = {t: (listed, d) for (t, listed, d) in universe}
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


# ---------------------------------------------------------------------------
# m7: EquityDataLoader + registry error-path tests.
# ---------------------------------------------------------------------------


def _write_universe_config(tmp_path, csv_rel=None, toml_body=None):
    """Helper: write a synthetic universe config + CSV under tmp_path."""
    csv = tmp_path / "u.csv"
    csv.write_text(
        "ticker,listed_at,delisted_at,name\nAAPL,,,Apple\n", encoding="utf-8"
    )
    csv_path = csv_rel or csv.as_posix()
    toml = tmp_path / "u.toml"
    toml.write_text(
        toml_body
        or (
            "[universe]\nname=\"u\"\n"
            f"universe_file='{csv_path}'\nsource=\"test\"\n"
        ),
        encoding="utf-8",
    )
    return toml


def test_loader_unknown_universe_raises_file_not_found():
    from equity.data.registry import get_universe_info

    with pytest.raises(FileNotFoundError, match="Unknown equity universe"):
        get_universe_info("definitely_not_a_universe_xyz")


def test_loader_config_lacking_universe_file_raises_value_error(tmp_path, monkeypatch):
    """A TOML that exists but has no [universe].universe_file -> ValueError."""
    from equity.data.registry import get_universe_info

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "bare.toml").write_text(
        '[universe]\nname = "bare"\n', encoding="utf-8"
    )
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", cfg_dir)
    with pytest.raises(ValueError, match="not a universe config"):
        get_universe_info("bare")


def test_loader_universe_csv_missing_required_column_raises(tmp_path, monkeypatch):
    csv = tmp_path / "u.csv"
    # Missing 'delisted_at'.
    csv.write_text(
        "ticker,listed_at,name\nAAPL,,Apple\n", encoding="utf-8"
    )
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    (cfg_dir / "u.toml").write_text(
        f"[universe]\nname=\"u\"\nuniverse_file='{csv.as_posix()}'\nsource=\"t\"\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", cfg_dir)
    with pytest.raises(ValueError, match="missing required column 'delisted_at'"):
        EquityDataLoader("u", "2024-01-01", "2024-12-31")


# ---------------------------------------------------------------------------
# m2: registry path-traversal name rejection.
# ---------------------------------------------------------------------------


def test_registry_rejects_path_traversal_universe_name():
    from equity.data.registry import _validate_universe_name

    for bad in ["..", "../x", "foo/bar", "foo\\bar", "a/../b"]:
        with pytest.raises(ValueError, match="must not contain"):
            _validate_universe_name(bad)
    # Empty name rejected.
    with pytest.raises(ValueError, match="non-empty"):
        _validate_universe_name("")
    # A clean name passes (no exception).
    _validate_universe_name("sp500")
    _validate_universe_name("sp500_2024")


def test_registry_path_traversal_via_get_universe_info(tmp_path, monkeypatch):
    """get_universe_info must reject names that would escape CONFIG_DIR."""
    from equity.data.registry import get_universe_info

    monkeypatch.setattr("equity.data.registry.CONFIG_DIR", tmp_path)
    with pytest.raises(ValueError, match="must not contain"):
        get_universe_info("../../etc/passwd")

