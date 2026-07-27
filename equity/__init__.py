"""
@module: equity
@depends: pandas, tomllib
@exports: EquityDataLoader, UniverseInfo, get_universe_info, list_universes
@paper_ref: N/A
@data_flow: universe config (configs/equity/*.toml) -> universe_file CSV -> alive-ticker tuples
"""

from equity.data.loader import EquityDataLoader
from equity.data.registry import UniverseInfo, get_universe_info, list_universes

# NOTE: equity has no separate pyproject; it is bundled into the stat-context
# wheel (packages.find include=["sce*","equity*"]). Do NOT declare a separate
# equity.__version__ -- the canonical version is stat-context's (pyproject.toml).

__all__ = [
    "EquityDataLoader",
    "UniverseInfo",
    "get_universe_info",
    "list_universes",
]
