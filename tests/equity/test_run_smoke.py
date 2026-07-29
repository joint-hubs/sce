"""
@module: tests.equity.test_run_smoke
@depends: equity.forecaster.run_smoke
@exports:
@data_flow: synthetic prices -> run_smoke -> predictions_hN.parquet + metadata.json

S5.5 single-fold smoke runner tests. Structural only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity.forecaster.config import (
    HorizonConfig,
    QuantileHeadParams,
    ResidualHeadParams,
    SectorHeadParams,
    SmokeConfig,
)
from equity.forecaster.metadata import build_metadata, config_hash, write_metadata
from equity.forecaster.run_smoke import run_smoke


def _prices(n: int = 60, n_tickers: int = 4) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01 16:00", periods=n, freq="D", tz="America/New_York")
    frames = []
    for t in range(n_tickers):
        rng = np.random.default_rng(50 + t)
        close = 100.0 * np.cumprod(1 + rng.normal(0.0004, 0.012, n))
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
                    "volume": rng.integers(1000, 9000, n).astype(float),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _sectors(n_tickers: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [f"TK{t}" for t in range(n_tickers)],
            "sector": ["Tech", "Tech", "Fin", "Energy"][:n_tickers],
        }
    )


@pytest.fixture(scope="module")
def smoke_result(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    # Keep horizons short so labels survive the 60-day panel; include one longer
    # horizon still short enough for n=60 (h=10 leaves 50 labels/ticker).
    cfg = SmokeConfig(
        horizon=HorizonConfig(
            horizons=(1, 5, 10),
            quantiles=(0.05, 0.5, 0.95),
            n_folds=3,
            seed=0,
        ),
        sector=SectorHeadParams(n_estimators=15, max_depth=2),
        residual=ResidualHeadParams(n_estimators=15, max_depth=2),
        quantile=QuantileHeadParams(n_estimators=15, max_depth=2, min_child_samples=5),
        test_frac=0.25,
        run_grade="exploratory",
        seed=0,
    )
    result = run_smoke(
        _prices(),
        sectors=_sectors(),
        config=cfg,
        out_dir=out,
        git_sha="deadbeef",
        created_at="2026-07-28T00:00:00Z",
    )
    return result


def test_smoke_writes_predictions_and_metadata(smoke_result) -> None:
    out = Path(smoke_result["out_dir"])
    assert (out / "metadata.json").is_file()
    for h in (1, 5, 10):
        path = out / f"predictions_h{h}.parquet"
        assert path.is_file(), path
        df = pd.read_parquet(path)
        assert f"pred_h{h}" in df.columns
        assert f"pred_sector_h{h}" in df.columns
        assert f"pred_resid_h{h}" in df.columns
        assert f"pred_h{h}_q05" in df.columns
        assert f"pred_h{h}_q50" in df.columns
        assert f"pred_h{h}_q95" in df.columns
        assert len(df) == smoke_result["n_test"]


def test_smoke_metadata_schema(smoke_result) -> None:
    meta = smoke_result["metadata"]
    for key in (
        "git_sha",
        "config_hash",
        "seed",
        "run_grade",
        "horizons",
        "quantiles",
        "created_at",
    ):
        assert key in meta, key
    assert meta["git_sha"] == "deadbeef"
    assert meta["run_grade"] == "exploratory"
    assert meta["seed"] == 0
    assert meta["horizons"] == [1, 5, 10]
    assert meta["created_at"] == "2026-07-28T00:00:00Z"


def test_smoke_ts_group_split_nonempty(smoke_result) -> None:
    assert smoke_result["n_train"] > 0
    assert smoke_result["n_test"] > 0


def test_metadata_writer_unit(tmp_path: Path) -> None:
    path = write_metadata(
        tmp_path,
        git_sha="abc",
        config_hash="hashhash",
        seed=1,
        run_grade="exploratory",
        horizons=(1, 5),
        quantiles=(0.05, 0.95),
        created_at="2020-01-01T00:00:00Z",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == build_metadata(
        git_sha="abc",
        config_hash="hashhash",
        seed=1,
        run_grade="exploratory",
        horizons=(1, 5),
        quantiles=(0.05, 0.95),
        created_at="2020-01-01T00:00:00Z",
    )
    # config_hash stability
    assert len(config_hash(SmokeConfig())) == 16
