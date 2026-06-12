"""
@module: sce.io
@depends: pandas, tomllib
@exports: DatasetInfo, ensure_dataset, get_dataset_info, list_datasets, load_dataset, save_dataset, verify_all_datasets, verify_dataset
@paper_ref: N/A
@data_flow: dataset name/config -> parquet path or download -> pandas.DataFrame
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
PARQUET_DIR = PROJECT_ROOT / "data" / "parquet"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "checksums.txt"
DOWNLOAD_SCRIPT = PROJECT_ROOT / "data" / "download.py"


@dataclass(frozen=True)
class DatasetInfo:
    """Metadata for a dataset known to the repository."""

    name: str
    path: Path
    source: str
    description: str
    remote_source: str | None
    checksum: str | None
    size_bytes: int | None
    exists_locally: bool


def _parse_manifest() -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    if not MANIFEST_PATH.exists():
        return entries

    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, checksum, size, source = [part.strip() for part in line.split("|", maxsplit=3)]
        entries[name] = {"checksum": checksum, "size": size, "source": source}
    return entries


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_dataset_reference(dataset: str | Path) -> tuple[str, Path, dict[str, Any]]:
    dataset_path = Path(dataset)
    if dataset_path.suffix == ".toml" or dataset_path.exists():
        config_path = dataset_path if dataset_path.is_absolute() else PROJECT_ROOT / dataset_path
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")
        payload = _load_toml(config_path)
        dataset_name = payload.get("dataset", {}).get("name") or config_path.stem
        return dataset_name, config_path, payload

    config_path = CONFIG_DIR / f"{dataset}.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Unknown dataset '{dataset}'. Expected config at {config_path}")
    return dataset, config_path, _load_toml(config_path)


def _dataset_info_from_config(config_path: Path) -> DatasetInfo:
    payload = _load_toml(config_path)
    dataset_cfg = payload.get("dataset", {})
    dataset_name = dataset_cfg.get("name", config_path.stem)
    relative_path = Path(dataset_cfg["path"])
    parquet_path = relative_path if relative_path.is_absolute() else PROJECT_ROOT / relative_path
    manifest_entry = _parse_manifest().get(parquet_path.name)
    remote_source = (
        None
        if manifest_entry is None or manifest_entry["source"] == "local"
        else manifest_entry["source"]
    )
    checksum = None if manifest_entry is None else manifest_entry["checksum"].replace("sha256:", "")
    size_bytes = None if manifest_entry is None else int(manifest_entry["size"])

    return DatasetInfo(
        name=dataset_name,
        path=parquet_path,
        source=dataset_cfg.get("source", "local"),
        description=dataset_cfg.get("description", ""),
        remote_source=remote_source,
        checksum=checksum,
        size_bytes=size_bytes,
        exists_locally=parquet_path.exists(),
    )


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def list_datasets() -> list[DatasetInfo]:
    """List datasets defined under the repo config directory.

    TOML files without a ``[dataset].path`` entry (e.g. shared report
    defaults) are not dataset configs and are skipped.
    """
    infos = []
    for path in CONFIG_DIR.glob("*.toml"):
        payload = _load_toml(path)
        if "path" not in payload.get("dataset", {}):
            continue
        infos.append(_dataset_info_from_config(path))
    return sorted(infos, key=lambda item: item.name)


def get_dataset_info(dataset: str | Path) -> DatasetInfo:
    """Return dataset metadata for a dataset name or config path."""
    _, config_path, _ = _resolve_dataset_reference(dataset)
    return _dataset_info_from_config(config_path)


def ensure_dataset(dataset: str | Path, force_download: bool = False) -> Path:
    """Ensure a dataset parquet exists locally, downloading it if configured as remote."""
    info = get_dataset_info(dataset)
    if info.exists_locally and not force_download:
        return info.path

    if info.remote_source is None:
        raise FileNotFoundError(f"Dataset not found locally: {info.path}")
    if not DOWNLOAD_SCRIPT.exists():
        raise FileNotFoundError(f"Dataset downloader not found: {DOWNLOAD_SCRIPT}")

    command = [sys.executable, str(DOWNLOAD_SCRIPT), "--dataset", info.path.name]
    if force_download:
        command.append("--force")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        error_text = result.stderr.strip() or result.stdout.strip() or "dataset download failed"
        raise RuntimeError(error_text)
    return info.path


def load_dataset(
    dataset: str | Path, force_download: bool = False, **read_kwargs: Any
) -> pd.DataFrame:
    """Load a configured dataset by name or config path."""
    dataset_path = ensure_dataset(dataset, force_download=force_download)
    return pd.read_parquet(dataset_path, **read_kwargs)


def save_dataset(df: pd.DataFrame, path: str | Path, **write_kwargs: Any) -> Path:
    """Save a DataFrame to parquet, creating parent directories if needed."""
    output_path = Path(path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False, **write_kwargs)
    return output_path


def verify_dataset(dataset: str | Path) -> bool:
    """Verify local size and checksum against the manifest when available."""
    info = get_dataset_info(dataset)
    if not info.path.exists() or info.checksum is None or info.size_bytes is None:
        return info.path.exists()
    return info.path.stat().st_size == info.size_bytes and _sha256(info.path) == info.checksum


def verify_all_datasets() -> dict[str, bool]:
    """Verify all configured datasets and return a name->status mapping."""
    return {info.name: verify_dataset(info.name) for info in list_datasets()}


__all__ = [
    "DatasetInfo",
    "ensure_dataset",
    "get_dataset_info",
    "list_datasets",
    "load_dataset",
    "save_dataset",
    "verify_all_datasets",
    "verify_dataset",
]
