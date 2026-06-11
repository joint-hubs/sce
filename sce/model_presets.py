"""
@module: sce.model_presets
@depends: tomllib
@exports: SUPPORTED_MODEL_TYPES, load_model_presets, resolve_model_presets, load_xgboost_presets, resolve_xgboost_presets
@data_flow: model type + TOML presets -> resolved parameter sets
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - fallback for older Python
    import tomli as tomllib  # type: ignore

_DEFAULT_PRESETS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "xgboost": {
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
    },
    "lightgbm": {
        "default": {
            "n_estimators": 300,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        "shallow": {
            "n_estimators": 250,
            "num_leaves": 15,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
        },
        "boosted": {
            "n_estimators": 500,
            "num_leaves": 63,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
    },
    "catboost": {
        "default": {
            "iterations": 300,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "RMSE",
        },
        "shallow": {
            "iterations": 250,
            "depth": 4,
            "learning_rate": 0.05,
            "loss_function": "RMSE",
        },
        "boosted": {
            "iterations": 500,
            "depth": 6,
            "learning_rate": 0.03,
            "loss_function": "RMSE",
        },
    },
    "ridge": {
        "default": {"alpha": 1.0},
        "light": {"alpha": 0.3},
        "strong": {"alpha": 10.0},
    },
    "random_forest": {
        "default": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        "shallow": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 2},
        "regularized": {
            "n_estimators": 400,
            "max_depth": 12,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
        },
    },
    "extra_trees": {
        "default": {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
        "shallow": {"n_estimators": 200, "max_depth": 8, "min_samples_leaf": 2},
        "regularized": {
            "n_estimators": 400,
            "max_depth": 12,
            "min_samples_leaf": 4,
            "max_features": "sqrt",
        },
    },
    "gradient_boosting": {
        "default": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.9},
        "shallow": {"n_estimators": 150, "max_depth": 2, "learning_rate": 0.05, "subsample": 0.9},
        "boosted": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8},
    },
}


def _normalize_model_type(model_type: str | None) -> str:
    normalized = (model_type or "xgboost").strip().lower()
    if normalized not in _DEFAULT_PRESETS:
        supported = ", ".join(sorted(_DEFAULT_PRESETS))
        raise ValueError(f"Unsupported model type '{model_type}'. Supported types: {supported}")
    return normalized


def _default_config_path(model_type: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "configs" / "models" / f"{model_type}.toml"


def _run_preset_key(model_type: str) -> str:
    return "xgboost_configs" if model_type == "xgboost" else f"{model_type}_configs"


def load_model_presets(
    model_type: str = "xgboost",
    config_path: Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Load preset definitions for a supported model type from TOML.

    Args:
        model_type: Supported downstream model type.
        config_path: Optional explicit path to presets TOML.

    Returns:
        Dict mapping preset name -> params.
    """
    model_type = _normalize_model_type(model_type)
    if config_path is None:
        config_path = _default_config_path(model_type)

    presets = deepcopy(_DEFAULT_PRESETS[model_type])

    if not config_path.exists():
        return presets

    with config_path.open("rb") as f:
        data = tomllib.load(f)

    presets.update({k: v for k, v in data.items() if isinstance(v, dict)})
    return presets


def resolve_model_presets(
    run_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    config_path: Path | None = None,
) -> Tuple[str, List[str], Dict[str, Dict[str, Any]]]:
    """Resolve model type, preset names, and params for a run.

    Args:
        run_cfg: `run` section from dataset config.
        model_cfg: `model` section from dataset config.
        config_path: Optional path to presets TOML.

    Returns:
        Tuple of (model_type, preset_names, preset_params).
    """
    model_type = _normalize_model_type(model_cfg.get("type") if isinstance(model_cfg, dict) else None)
    presets = load_model_presets(model_type, config_path)

    preset_names = None
    if isinstance(run_cfg, dict):
        preset_names = run_cfg.get(_run_preset_key(model_type)) or run_cfg.get("model_configs")

    if preset_names:
        names = [n for n in preset_names if n in presets] or ["default"]
        overrides = {k: v for k, v in model_cfg.items() if k != "type"}
        if overrides:
            presets = presets.copy()
            for name in names:
                presets[name] = {**presets[name], **overrides}
        return model_type, names, presets

    if model_cfg:
        presets = presets.copy()
        presets["default"] = {
            **presets["default"],
            **{k: v for k, v in model_cfg.items() if k != "type"},
        }
        return model_type, ["default"], presets

    return model_type, ["default"], presets


def load_xgboost_presets(config_path: Path | None = None) -> Dict[str, Dict[str, Any]]:
    """Load XGBoost preset definitions from TOML."""
    return load_model_presets("xgboost", config_path)


def resolve_xgboost_presets(
    run_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    config_path: Path | None = None,
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Backward-compatible wrapper for XGBoost preset resolution."""
    _, names, presets = resolve_model_presets(
        run_cfg,
        {**(model_cfg or {}), "type": "xgboost"},
        config_path,
    )
    return names, presets

SUPPORTED_MODEL_TYPES: Tuple[str, ...] = tuple(sorted(_DEFAULT_PRESETS))
