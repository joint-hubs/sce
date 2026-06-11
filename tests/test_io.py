"""
@module: tests.test_io
@depends: sce.io
@exports:
@data_flow: config/manifest -> dataset metadata -> parquet load/save
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sce.io import get_dataset_info, list_datasets, load_dataset, save_dataset, verify_dataset


PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_DATA_PATH = PROJECT_ROOT / "data" / "parquet" / "rental_poland_short.parquet"


def test_list_datasets_includes_known_configs():
    names = {info.name for info in list_datasets()}
    assert "rental_poland_short" in names
    assert "sales_uae_transactions" in names


def test_get_dataset_info_uses_manifest_metadata():
    info = get_dataset_info("rental_uae_contracts")

    assert info.name == "rental_uae_contracts"
    assert info.remote_source is not None
    assert info.path.name == "rental_uae_contracts.parquet"


@pytest.mark.skipif(not LOCAL_DATA_PATH.exists(), reason="Local parquet file not available")
def test_load_dataset_reads_local_dataset():
    df = load_dataset("rental_poland_short")

    assert not df.empty
    assert "price_PLN_per_night" in df.columns


def test_save_dataset_writes_parquet(tmp_path: Path):
    df = pd.DataFrame({"city": ["a", "b"], "price": [1.0, 2.0]})

    output_path = save_dataset(df, tmp_path / "saved.parquet")

    assert output_path.exists()
    restored = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(restored, df)


@pytest.mark.skipif(not LOCAL_DATA_PATH.exists(), reason="Local parquet file not available")
def test_verify_dataset_matches_manifest_for_local_data():
    assert verify_dataset("rental_poland_short") is True