"""
@module: equity.forecaster.metadata
@depends: hashlib, json, pathlib, subprocess
@exports: write_metadata, collect_git_sha, config_hash, build_metadata
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §7 (run metadata)
@data_flow: knobs -> metadata dict -> metadata.json under out_dir

Standalone metadata writer for S5 smoke runs. Deliberately does NOT import
``scripts.run._collect_run_metadata`` (tightly coupled to the SCE experiment
runner). Schema: ``{git_sha, config_hash, seed, run_grade, horizons, quantiles,
created_at}``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


def collect_git_sha(cwd: Optional[Path] = None) -> str:
    """Return the short HEAD sha, or ``"unknown"`` if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd) if cwd is not None else None,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def config_hash(config: Any) -> str:
    """Stable sha256 prefix over a dataclass / mapping / JSON-able object."""
    if is_dataclass(config) and not isinstance(config, type):
        payload: Any = asdict(config)
    elif isinstance(config, Mapping):
        payload = dict(config)
    else:
        payload = config
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def build_metadata(
    *,
    git_sha: str,
    config_hash: str,
    seed: int,
    run_grade: str,
    horizons: Sequence[int],
    quantiles: Sequence[float],
    created_at: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Assemble the metadata dict (pure; no I/O)."""
    if created_at is None:
        created_at = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta: dict[str, Any] = {
        "git_sha": git_sha,
        "config_hash": config_hash,
        "seed": int(seed),
        "run_grade": str(run_grade),
        "horizons": [int(h) for h in horizons],
        "quantiles": [float(q) for q in quantiles],
        "created_at": created_at,
    }
    if extra:
        for k, v in extra.items():
            if k not in meta:
                meta[k] = v
    return meta


def write_metadata(
    out_dir: str | Path,
    *,
    git_sha: str,
    config_hash: str,
    seed: int,
    run_grade: str,
    horizons: Sequence[int],
    quantiles: Sequence[float],
    created_at: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
    filename: str = "metadata.json",
) -> Path:
    """Write ``metadata.json`` under ``out_dir`` and return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = build_metadata(
        git_sha=git_sha,
        config_hash=config_hash,
        seed=seed,
        run_grade=run_grade,
        horizons=horizons,
        quantiles=quantiles,
        created_at=created_at,
        extra=extra,
    )
    path = out / filename
    path.write_text(
        json.dumps(meta, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path
