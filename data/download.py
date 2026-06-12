"""
@module: data.download
@depends:
@exports: download_dataset, download_all, verify_all, parse_source
@data_flow: manifest -> remote provider -> local parquet -> checksum verify

Data download script for manifest-backed datasets.

Usage:
    python data/download.py --dataset rental_uae_contracts.parquet
    python data/download.py --all
    python data/download.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import NamedTuple

DATA_DIR = Path(__file__).parent
MANIFEST_PATH = DATA_DIR / "manifests" / "checksums.txt"
PARQUET_DIR = DATA_DIR / "parquet"


class DatasetEntry(NamedTuple):
    name: str
    checksum: str
    size: int
    source: str


class SourceSpec(NamedTuple):
    provider: str
    resource_type: str
    resource: str
    file_name: str
    raw_source: str


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


def parse_source(source: str) -> SourceSpec:
    """Parse a manifest source into a provider-aware spec."""
    if source == "local":
        return SourceSpec("local", "", "", "", source)

    if source.startswith(("http://", "https://")):
        return SourceSpec("http", "", source, "", source)

    if not source.startswith("kaggle://"):
        raise ValueError(
            "Unsupported dataset source. Expected local, http(s), or kaggle://..."
        )

    path = source[len("kaggle://") :]
    parts = [part for part in path.split("/") if part]
    if len(parts) < 3:
        raise ValueError(
            "Kaggle sources must use kaggle://datasets/<owner>/<dataset>/<file> "
            "or kaggle://competitions/<competition>/<file>"
        )

    resource_type = parts[0]
    file_name = parts[-1]
    resource = "/".join(parts[1:-1])
    if resource_type not in {"datasets", "competitions"} or not resource or not file_name:
        raise ValueError(
            "Kaggle sources must use kaggle://datasets/<owner>/<dataset>/<file> "
            "or kaggle://competitions/<competition>/<file>"
        )

    return SourceSpec("kaggle", resource_type, resource, file_name, source)


def download_file(url: str, dest: Path, expected_size: int, retries: int = 3) -> None:
    """Download file with progress bar and retry logic."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Dataset downloads require the 'data' extra: pip install stat-context[data]"
        ) from exc

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


def _extract_from_archive(archive_path: Path, requested_name: str, output_dir: Path) -> Path:
    with zipfile.ZipFile(archive_path) as archive:
        requested_basename = Path(requested_name).name
        for member in archive.namelist():
            if Path(member).name == requested_basename:
                archive.extract(member, output_dir)
                return output_dir / member

    raise RuntimeError(f"Could not find '{requested_name}' inside {archive_path.name}")


def _find_downloaded_file(download_dir: Path, requested_name: str) -> Path:
    requested_basename = Path(requested_name).name

    direct_match = download_dir / requested_name
    if direct_match.exists():
        return direct_match

    for candidate in download_dir.rglob("*"):
        if candidate.is_file() and candidate.name == requested_basename:
            return candidate

    for archive_path in download_dir.glob("*.zip"):
        return _extract_from_archive(archive_path, requested_name, download_dir)

    raise RuntimeError(f"Downloaded artifact '{requested_name}' was not found")


def _resolve_kaggle_command() -> list[str]:
    for name in ("kaggle.exe", "kaggle"):
        candidate = Path(sys.executable).with_name(name)
        if candidate.exists():
            return [str(candidate)]
    return [sys.executable, "-m", "kaggle.cli"]


def download_kaggle_file(spec: SourceSpec, dest: Path) -> None:
    """Download a single file from Kaggle via the official CLI."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        command = _resolve_kaggle_command()
        if spec.resource_type == "competitions":
            command.extend(
                [
                    "competitions",
                    "download",
                    "-c",
                    spec.resource,
                    "-f",
                    spec.file_name,
                    "-p",
                    tmp_dir,
                    "-q",
                ]
            )
        else:
            command.extend(
                [
                    "datasets",
                    "download",
                    "-d",
                    spec.resource,
                    "-f",
                    spec.file_name,
                    "-p",
                    tmp_dir,
                    "-q",
                ]
            )

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            error_text = result.stderr.strip() or result.stdout.strip() or "unknown Kaggle error"
            raise RuntimeError(error_text)

        downloaded = _find_downloaded_file(Path(tmp_dir), spec.file_name)
        shutil.move(str(downloaded), dest)


def download_from_source(entry: DatasetEntry, dest: Path) -> None:
    """Download a dataset from the provider declared in the manifest."""
    spec = parse_source(entry.source)
    if spec.provider == "http":
        download_file(spec.resource, dest, entry.size)
        return

    if spec.provider == "kaggle":
        download_kaggle_file(spec, dest)
        return

    raise RuntimeError(f"Dataset '{entry.name}' is not downloadable from source '{entry.source}'")


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
    try:
        download_from_source(entry, dest)
    except RuntimeError as exc:
        print(f"Error: Failed to download '{name}': {exc}")
        dest.unlink(missing_ok=True)
        return False

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
