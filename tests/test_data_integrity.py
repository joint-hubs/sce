"""
@module: tests.test_data_integrity
@depends:
@exports:
@data_flow: manifests -> parquet files -> schema/integrity checks
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pandas.api.types as ptypes
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "checksums.txt"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"

# Check if local parquet files exist (they may be gitignored)
LOCAL_PARQUETS_EXIST = any(PARQUET_DIR.glob("*.parquet")) if PARQUET_DIR.exists() else False


LOCAL_DATASETS = {
    "rental_poland_short.parquet": {
        "expected_columns": [
            "city",
            "room_type",
            "property_type",
            "capacity",
            "is_superhost",
            "host_rating",
            "host_review_count",
            "rating_cleanliness",
            "review_count",
            "price_display",
            "price_PLN_per_night",
        ],
        "numeric_columns": [
            "capacity",
            "host_rating",
            "host_review_count",
            "rating_cleanliness",
            "review_count",
            "price_PLN_per_night",
        ],
        "non_negative_columns": [
            "capacity",
            "host_rating",
            "host_review_count",
            "rating_cleanliness",
            "review_count",
            "price_PLN_per_night",
        ],
    },
    "rental_poland_long.parquet": {
        "expected_columns": [
            "city",
            "district",
            "region",
            "subdistrict",
            "area_sqm",
            "price",
            "price_currency",
            "price_per_sqm",
            "rooms",
            "estate_type",
            "transaction_type",
            "floor_number",
            "development_estate_type",
            "development_floor",
        ],
        "numeric_columns": [
            "area_sqm",
            "price",
            "price_per_sqm",
            "rooms",
            "development_floor",
        ],
        "non_negative_columns": [
            "area_sqm",
            "price",
            "price_per_sqm",
            "rooms",
            "development_floor",
        ],
    },
}

REMOTE_DATASETS = {
    "rental_uae_contracts.parquet",
    "sales_uae_transactions.parquet",
}


def _parse_manifest(path: Path) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            raise ValueError(f"Invalid manifest line: {line}")
        name, checksum, size, source = parts
        entries[name] = {
            "checksum": checksum,
            "size": size,
            "source": source,
        }
    return entries


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_manifest_has_local_entries():
    manifest = _parse_manifest(MANIFEST_PATH)
    for name in LOCAL_DATASETS:
        assert name in manifest, f"Missing manifest entry for {name}"
        entry = manifest[name]
        assert entry["source"] == "local"
        assert entry["checksum"].startswith("sha256:")
        assert entry["checksum"] != "sha256:pending"
        assert int(entry["size"]) > 0


def test_manifest_has_remote_entries():
    manifest = _parse_manifest(MANIFEST_PATH)
    for name in REMOTE_DATASETS:
        assert name in manifest, f"Missing manifest entry for {name}"
        entry = manifest[name]
        assert entry["source"].startswith("https://")
        assert entry["checksum"].startswith("sha256:")
        assert entry["checksum"] != "sha256:pending"
        assert int(entry["size"]) > 0


@pytest.mark.skipif(not LOCAL_PARQUETS_EXIST, reason="Local parquet files not available")
def test_local_parquet_checksums_match_manifest():
    manifest = _parse_manifest(MANIFEST_PATH)
    for name in LOCAL_DATASETS:
        parquet_path = PARQUET_DIR / name
        if not parquet_path.exists():
            pytest.skip(f"Parquet file not found: {name}")
        entry = manifest[name]
        expected_checksum = entry["checksum"].split("sha256:")[-1]
        expected_size = int(entry["size"])
        assert parquet_path.stat().st_size == expected_size
        assert _sha256(parquet_path) == expected_checksum


@pytest.mark.skipif(not LOCAL_PARQUETS_EXIST, reason="Local parquet files not available")
def test_local_parquet_schema_is_expected():
    """Verify all expected columns are present (order may vary due to parquet)."""
    for name, rules in LOCAL_DATASETS.items():
        parquet_path = PARQUET_DIR / name
        if not parquet_path.exists():
            pytest.skip(f"Parquet file not found: {name}")
        df = pd.read_parquet(parquet_path)
        expected = set(rules["expected_columns"])
        actual = set(df.columns)
        assert actual == expected, f"{name}: columns mismatch. Expected {expected}, got {actual}"


@pytest.mark.skipif(not LOCAL_PARQUETS_EXIST, reason="Local parquet files not available")
def test_local_parquet_numeric_columns_are_numeric():
    for name, rules in LOCAL_DATASETS.items():
        parquet_path = PARQUET_DIR / name
        if not parquet_path.exists():
            pytest.skip(f"Parquet file not found: {name}")
        df = pd.read_parquet(parquet_path)
        for col in rules["numeric_columns"]:
            assert col in df.columns
            assert ptypes.is_numeric_dtype(df[col]), f"{name}:{col} is not numeric"


@pytest.mark.skipif(not LOCAL_PARQUETS_EXIST, reason="Local parquet files not available")
def test_local_parquet_values_non_negative():
    for name, rules in LOCAL_DATASETS.items():
        parquet_path = PARQUET_DIR / name
        if not parquet_path.exists():
            pytest.skip(f"Parquet file not found: {name}")
        df = pd.read_parquet(parquet_path)
        for col in rules["non_negative_columns"]:
            series = df[col].dropna()
            if not series.empty:
                assert (series >= 0).all(), f"{name}:{col} has negative values"
