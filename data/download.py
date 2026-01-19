"""
@module: data.download
@depends:
@exports: download_dataset, download_all, verify_integrity
@data_flow: manifest -> HF Hub -> local parquet -> checksum verify

Data download script for remote datasets.

Usage:
    python data/download.py --dataset rental_uae_contracts
    python data/download.py --all
    python data/download.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import NamedTuple

import requests
from tqdm import tqdm


DATA_DIR = Path(__file__).parent
MANIFEST_PATH = DATA_DIR / "manifests" / "checksums.txt"
PARQUET_DIR = DATA_DIR / "parquet"


class DatasetEntry(NamedTuple):
    name: str
    checksum: str
    size: int
    source: str


def parse_manifest() -> dict[str, DatasetEntry]:
    """Parse checksums.txt manifest file."""
    entries: dict[str, DatasetEntry] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 4:
            continue
        name, checksum, size, source = parts
        entries[name] = DatasetEntry(
            name=name,
            checksum=checksum.replace("sha256:", ""),
            size=int(size),
            source=source,
        )
    return entries


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file(path: Path, entry: DatasetEntry) -> bool:
    """Verify file exists with correct size and checksum."""
    if not path.exists():
        return False
    if path.stat().st_size != entry.size:
        return False
    return sha256_file(path) == entry.checksum


def download_file(url: str, dest: Path, expected_size: int, retries: int = 3) -> None:
    """Download file with progress bar and retry logic."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total = expected_size or int(response.headers.get("content-length", 0))

            with dest.open("wb") as f, tqdm(
                desc=dest.name,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    pbar.update(len(chunk))
            return

        except (requests.RequestException, IOError) as e:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url} after {retries} attempts: {e}")
            print(f"  Retry {attempt}/{retries} after error: {e}")


def download_dataset(name: str, force: bool = False) -> bool:
    """Download a single dataset by name."""
    manifest = parse_manifest()

    if name not in manifest:
        print(f"Error: Unknown dataset '{name}'")
        print(f"Available: {', '.join(manifest.keys())}")
        return False

    entry = manifest[name]

    if entry.source == "local":
        print(f"'{name}' is a local dataset (no download needed)")
        return True

    dest = PARQUET_DIR / name

    if not force and verify_file(dest, entry):
        print(f"'{name}' already exists and is valid (use --force to re-download)")
        return True

    print(f"Downloading '{name}' from {entry.source}...")
    download_file(entry.source, dest, entry.size)

    if not verify_file(dest, entry):
        print(f"Error: Checksum mismatch for '{name}'!")
        dest.unlink(missing_ok=True)
        return False

    print(f"✓ '{name}' downloaded and verified")
    return True


def download_all(force: bool = False) -> bool:
    """Download all remote datasets."""
    manifest = parse_manifest()
    success = True

    for name, entry in manifest.items():
        if entry.source == "local":
            continue
        if not download_dataset(name, force=force):
            success = False

    return success


def verify_all() -> bool:
    """Verify integrity of all datasets."""
    manifest = parse_manifest()
    all_valid = True

    for name, entry in manifest.items():
        path = PARQUET_DIR / name

        if entry.source == "local":
            if not path.exists():
                print(f"✗ {name}: MISSING (local dataset)")
                all_valid = False
            elif not verify_file(path, entry):
                print(f"✗ {name}: INVALID (checksum mismatch)")
                all_valid = False
            else:
                print(f"✓ {name}: OK")
        else:
            if not path.exists():
                print(f"○ {name}: NOT DOWNLOADED (remote)")
            elif not verify_file(path, entry):
                print(f"✗ {name}: INVALID (checksum mismatch)")
                all_valid = False
            else:
                print(f"✓ {name}: OK")

    return all_valid


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and verify SCE datasets")
    parser.add_argument("--dataset", "-d", help="Dataset name to download")
    parser.add_argument("--all", "-a", action="store_true", help="Download all remote datasets")
    parser.add_argument("--verify", "-v", action="store_true", help="Verify all dataset checksums")
    parser.add_argument("--force", "-f", action="store_true", help="Force re-download")

    args = parser.parse_args()

    if args.verify:
        return 0 if verify_all() else 1

    if args.all:
        return 0 if download_all(force=args.force) else 1

    if args.dataset:
        return 0 if download_dataset(args.dataset, force=args.force) else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
