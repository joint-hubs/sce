#!/usr/bin/env python3
"""
Download datasets from Hugging Face Hub.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --dataset rental_uae_contracts
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import requests

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "parquet"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "checksums.txt"


def sha256_file(path: Path) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_manifest() -> dict[str, dict[str, str]]:
    """Parse manifest file to get dataset metadata."""
    datasets = {}
    with MANIFEST_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                name, checksum, size, source = parts[:4]
                if source.startswith("http"):
                    datasets[name] = {
                        "checksum": checksum.replace("sha256:", ""),
                        "size": int(size),
                        "url": source,
                    }
    return datasets


def download_file(url: str, dest: Path, expected_size: int | None = None) -> None:
    """Download a file with progress bar."""
    print(f"Downloading {dest.name}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    if expected_size and total_size and total_size != expected_size:
        print(f"  Warning: Expected {expected_size} bytes, got {total_size} bytes")
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if HAS_TQDM:
        with dest.open("wb") as f, tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    else:
        # Simple progress without tqdm
        downloaded = 0
        with dest.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        print(f"  Progress: {pct:.1f}% ({downloaded}/{total_size} bytes)", end="\r")
        if total_size > 0:
            print()  # New line after progress
    
    print(f"  Downloaded to {dest}")


def verify_checksum(path: Path, expected: str) -> bool:
    """Verify file checksum."""
    actual = sha256_file(path)
    if actual != expected:
        print("  ✗ Checksum mismatch!")
        print(f"    Expected: {expected}")
        print(f"    Got:      {actual}")
        return False
    print("  ✓ Checksum verified")
    return True


def download_dataset(name: str, metadata: dict[str, str], force: bool = False) -> bool:
    """Download a single dataset."""
    dest = DATA_DIR / name
    
    # Check if already exists
    if dest.exists() and not force:
        print(f"Checking {name}...")
        if verify_checksum(dest, metadata["checksum"]):
            print("  ✓ Already downloaded and verified")
            return True
        else:
            print("  Checksum failed, re-downloading...")
    
    # Download
    try:
        download_file(metadata["url"], dest, metadata.get("size"))
        
        # Verify
        if not verify_checksum(dest, metadata["checksum"]):
            dest.unlink()
            return False
        
        return True
    except Exception as e:
        print(f"  ✗ Error downloading {name}: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(description="Download datasets from Hugging Face")
    parser.add_argument(
        "--dataset",
        help="Specific dataset to download (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if file exists",
    )
    args = parser.parse_args()
    
    datasets = parse_manifest()
    
    if args.dataset:
        if args.dataset not in datasets:
            print(f"Error: Dataset '{args.dataset}' not found in manifest")
            print(f"Available datasets: {', '.join(datasets.keys())}")
            return 1
        datasets = {args.dataset: datasets[args.dataset]}
    
    print(f"Found {len(datasets)} remote dataset(s) to download\n")
    
    success_count = 0
    for name, metadata in datasets.items():
        if download_dataset(name, metadata, args.force):
            success_count += 1
        print()
    
    print(f"Downloaded {success_count}/{len(datasets)} datasets successfully")
    return 0 if success_count == len(datasets) else 1


if __name__ == "__main__":
    exit(main())
