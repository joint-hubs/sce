"""
@module: tests.equity.test_survivorship_check
@depends: equity.diagnostics.survivorship_check
@exports:
@data_flow: configs/equity/sp500_universe.csv -> run_survivorship_check -> pass/fail

S4.5 survivorship-bias seed check unit tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from equity.diagnostics.survivorship_check import (
    run_survivorship_check,
)


def test_sp500_seed_passes_default_min() -> None:
    result = run_survivorship_check("sp500")
    assert result["pass"] is True
    assert result["n_delisted"] >= 13
    assert result["n_required"] == 13
    assert result["n_total"] >= result["n_delisted"]
    assert result["missing_tickers"] == []
    assert result["universe"] == "sp500"


def test_min_delisted_raised_fails() -> None:
    result = run_survivorship_check("sp500", min_delisted=50)
    assert result["pass"] is False
    assert result["n_delisted"] < 50
    assert result["n_required"] == 50


def test_bad_universe_name_raises() -> None:
    with pytest.raises((FileNotFoundError, ValueError)):
        run_survivorship_check("does_not_exist_universe_xyz")


def test_path_to_csv(tmp_path: Path) -> None:
    csv = tmp_path / "toy_universe.csv"
    csv.write_text(
        "ticker,listed_at,delisted_at,name\n"
        "AAA,,,Alive\n"
        "BBB,,2020-01-01,Dead One\n"
        "CCC,,2021-06-15,Dead Two\n",
        encoding="utf-8",
    )
    ok = run_survivorship_check(csv, min_delisted=2)
    assert ok["pass"] is True
    assert ok["n_delisted"] == 2
    assert ok["n_total"] == 3

    fail = run_survivorship_check(csv, min_delisted=3)
    assert fail["pass"] is False


def test_cli_sp500(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from equity.diagnostics import survivorship_check as mod

    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        ["survivorship_check", "--universe", "sp500", "--min-delisted", "13"],
    )
    rc = mod.main()
    assert rc == 0
