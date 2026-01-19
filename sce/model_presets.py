"""
@module: sce.model_presets
@depends: tomllib
@exports: load_xgboost_presets, resolve_xgboost_presets
@data_flow: toml -> preset_map -> resolved_presets
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib  # type: ignore

_DEFAULT_PRESETS: Dict[str, Dict[str, Any]] = {
    "default": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "shallow": {
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "boosted": {
        "n_estimators": 300,
        "max_depth": 5,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
}


def load_xgboost_presets(config_path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Load XGBoost preset definitions from TOML.

    Args:
        config_path: Optional explicit path to presets TOML.

    Returns:
        Dict mapping preset name -> params.
    """
    if config_path is None:
        repo_root = Path(__file__).resolve().parents[2]
        config_path = repo_root / "configs" / "models" / "xgboost.toml"

    if not config_path.exists():
        return _DEFAULT_PRESETS.copy()

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    presets = _DEFAULT_PRESETS.copy()
    presets.update({k: v for k, v in data.items() if isinstance(v, dict)})
    return presets


def resolve_xgboost_presets(
    run_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    config_path: Path | None = None,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Resolve preset names + params for a run.

    Args:
        run_cfg: `run` section from dataset config.
        model_cfg: `model` section from dataset config (legacy single config).
        config_path: Optional path to presets TOML.

    Returns:
        Tuple of (preset_names, preset_params).
    """
    presets = load_xgboost_presets(config_path)
    preset_names = run_cfg.get("xgboost_configs") if isinstance(run_cfg, dict) else None

    if preset_names:
        names = [n for n in preset_names if n in presets]
        return (names or ["default"], presets)

    # Fallback: single config from legacy [model]
    if model_cfg:
        presets = presets.copy()
        presets["default"] = {
            "n_estimators": model_cfg.get("n_estimators", presets["default"]["n_estimators"]),
            "max_depth": model_cfg.get("max_depth", presets["default"]["max_depth"]),
            "learning_rate": model_cfg.get("learning_rate", presets["default"]["learning_rate"]),
            "subsample": model_cfg.get("subsample", presets["default"].get("subsample", 0.8)),
            "colsample_bytree": model_cfg.get(
                "colsample_bytree", presets["default"].get("colsample_bytree", 0.8)
            ),
        }
        return ["default"], presets

    return ["default"], presets
