"""
@module: equity.data.registry
@depends: tomllib
@exports: UniverseInfo, list_universes, get_universe_info
@paper_ref: N/A
@data_flow: configs/equity/*.toml -> UniverseInfo metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


# equity/data/registry.py -> equity/data -> equity -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs" / "equity"


@dataclass(frozen=True)
class UniverseInfo:
    """Metadata for an equity universe defined under ``configs/equity/``."""

    name: str
    path: Path
    universe_file: Path
    description: str
    source: str
    exists_locally: bool


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _universe_info_from_config(config_path: Path) -> UniverseInfo:
    payload = _load_toml(config_path)
    universe_cfg = payload.get("universe", {})
    universe_name = universe_cfg.get("name", config_path.stem)
    relative_file = Path(universe_cfg["universe_file"])
    universe_file = (
        relative_file if relative_file.is_absolute() else PROJECT_ROOT / relative_file
    )
    return UniverseInfo(
        name=universe_name,
        path=config_path,
        universe_file=universe_file,
        description=universe_cfg.get("description", ""),
        source=universe_cfg.get("source", "local"),
        exists_locally=universe_file.exists(),
    )


def list_universes() -> list[UniverseInfo]:
    """List equity universes defined under ``configs/equity/``.

    TOML files without a ``[universe].universe_file`` entry are skipped
    (they are not universe configs).
    """
    infos: list[UniverseInfo] = []
    if not CONFIG_DIR.exists():
        return infos
    for path in CONFIG_DIR.glob("*.toml"):
        payload = _load_toml(path)
        if "universe_file" not in payload.get("universe", {}):
            continue
        infos.append(_universe_info_from_config(path))
    return sorted(infos, key=lambda item: item.name)


def get_universe_info(name: str) -> UniverseInfo:
    """Return universe metadata for a universe name (e.g. ``"sp500"``)."""
    config_path = CONFIG_DIR / f"{name}.toml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Unknown equity universe '{name}'. Expected config at {config_path}"
        )
    payload = _load_toml(config_path)
    if "universe_file" not in payload.get("universe", {}):
        raise ValueError(
            f"Config {config_path} is not a universe config "
            f"(missing [universe].universe_file)."
        )
    return _universe_info_from_config(config_path)


__all__ = ["UniverseInfo", "get_universe_info", "list_universes"]
