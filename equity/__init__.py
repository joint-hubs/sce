"""
@module: equity
@depends: pandas, tomllib
@exports: EquityDataLoader, UniverseInfo, get_universe_info, list_universes
@paper_ref: N/A
@data_flow: universe config (configs/equity/*.toml) -> universe_file CSV -> alive-ticker tuples
"""

from equity.data.loader import EquityDataLoader
from equity.data.registry import UniverseInfo, get_universe_info, list_universes

__version__ = "0.1.0"
__all__ = [
    "EquityDataLoader",
    "UniverseInfo",
    "get_universe_info",
    "list_universes",
]
