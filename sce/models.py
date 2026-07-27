"""
@module: sce.models
@depends: numpy, pandas, sklearn, xgboost, lightgbm, catboost
@exports: build_model, extract_feature_importance, get_model_label, model_supports_gpu
@data_flow: model_type + params -> estimator -> feature_importance
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _inf_to_nan(X: Any) -> Any:
    """Replace +/-inf with NaN so the downstream imputer can fill them.

    SCE ratio/z-score features can be infinite (e.g. y/median when median=0, or
    a z-score when the group std is 0). Tree models tolerate inf, but linear
    models and StandardScaler turn it into garbage. Routing inf through NaN lets
    the median imputer neutralize it.
    """
    arr = np.asarray(X, dtype=float)
    return np.where(np.isinf(arr), np.nan, arr)


# Wide log-spaced grid for RidgeCV. SCE feature blocks are high-dimensional and
# strongly collinear (many quantiles of the same group), so on small datasets
# (n < p) a fixed alpha=1 underregularizes and predictions explode. Cross-
# validating alpha per fit keeps the linear baseline stable.
_RIDGE_ALPHA_GRID = np.logspace(-1.0, 4.0, 20)

_MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient Boosting",
}

_GPU_CAPABLE_MODELS = {"xgboost", "lightgbm", "catboost"}


def get_model_label(model_type: str) -> str:
    """Return a human-readable label for a model type."""
    return _MODEL_LABELS.get(model_type, model_type.replace("_", " ").title())


def model_supports_gpu(model_type: str) -> bool:
    """Return whether the given model type can use a GPU backend."""
    return model_type.lower() in _GPU_CAPABLE_MODELS


def _filtered_params(
    estimator: Any, params: dict[str, Any], defaults: dict[str, Any]
) -> dict[str, Any]:
    valid_params = estimator.get_params(deep=False)
    resolved = {key: value for key, value in defaults.items() if key in valid_params}
    resolved.update({key: value for key, value in params.items() if key in valid_params})
    return resolved


def _extract_runtime_hints(params: dict[str, Any]) -> tuple[dict[str, Any], bool, int | None]:
    resolved = dict(params)
    use_gpu = bool(resolved.pop("use_gpu", False))
    gpu_device_id = resolved.pop("gpu_device_id", None)
    return resolved, use_gpu, gpu_device_id


def _wrap_sklearn(estimator: Any, *, scale: bool = False) -> Pipeline:
    """Wrap a NaN-intolerant sklearn estimator in an imputing pipeline.

    Ridge, RandomForest, ExtraTrees and GradientBoosting (the pure-sklearn
    estimators) reject NaN inputs, unlike the gradient-boosting libraries which
    handle missing values natively. SCE feature blocks frequently contain NaNs
    for small/empty groups, so we median-impute before fitting. ``scale=True``
    additionally standardizes features for scale-sensitive linear models
    (without it Ridge can blow up by hundreds of percent on raw-scale targets).

    ``keep_empty_features=True`` keeps all-NaN columns (filled with the impute
    statistic) so the feature count stays aligned with the input — required for
    feature-importance extraction. Needs scikit-learn >= 1.2.
    """
    steps = [
        ("sanitize", FunctionTransformer(_inf_to_nan, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def _final_estimator(model: Any) -> Any:
    """Return the underlying estimator, unwrapping a Pipeline if present."""
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def _normalize_catboost_params(params: dict[str, Any]) -> dict[str, Any]:
    """Map common generic aliases onto CatBoost's preferred parameter names."""
    resolved = dict(params)

    if "iterations" not in resolved and "n_estimators" in resolved:
        resolved["iterations"] = resolved.pop("n_estimators")
    else:
        resolved.pop("n_estimators", None)

    if "depth" not in resolved and "max_depth" in resolved:
        resolved["depth"] = resolved.pop("max_depth")
    else:
        resolved.pop("max_depth", None)

    if "random_seed" not in resolved and "random_state" in resolved:
        resolved["random_seed"] = resolved.pop("random_state")
    else:
        resolved.pop("random_state", None)

    if "rsm" not in resolved and "colsample_bytree" in resolved:
        resolved["rsm"] = resolved.pop("colsample_bytree")
    else:
        resolved.pop("colsample_bytree", None)

    return resolved


def build_model(model_type: str, params: dict[str, Any] | None = None) -> Any:
    """Instantiate a supported downstream regression model."""
    model_type = model_type.lower()
    params, use_gpu, gpu_device_id = _extract_runtime_hints(params or {})

    if model_type == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError("xgboost is required when model.type = 'xgboost'") from exc
        estimator = XGBRegressor()
        defaults = {"random_state": 42, "n_jobs": -1, "verbosity": 0}
        if use_gpu:
            defaults["device"] = "cuda" if gpu_device_id is None else f"cuda:{int(gpu_device_id)}"
            defaults.setdefault("tree_method", "hist")
        resolved = _filtered_params(
            estimator,
            params,
            defaults,
        )
        return XGBRegressor(**resolved)

    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError("lightgbm is required when model.type = 'lightgbm'") from exc

        resolved = {"random_state": 42, "n_jobs": -1, "verbosity": -1, **params}
        if use_gpu:
            resolved.setdefault("device", "gpu")
            if gpu_device_id is not None:
                resolved.setdefault("gpu_device_id", int(gpu_device_id))
        return LGBMRegressor(**resolved)

    if model_type == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError("catboost is required when model.type = 'catboost'") from exc

        resolved = {"random_seed": 42, "verbose": False, **_normalize_catboost_params(params)}
        if use_gpu:
            resolved.setdefault("task_type", "GPU")
            if gpu_device_id is not None:
                resolved.setdefault("devices", str(int(gpu_device_id)))
        return CatBoostRegressor(**resolved)

    if model_type == "ridge":
        # RidgeCV auto-selects alpha over a wide grid (efficient leave-one-out),
        # which keeps the linear model stable on wide collinear SCE features.
        # Presets pass a scalar `alpha`; it is not a RidgeCV param and is dropped
        # by _filtered_params, so the cross-validated grid takes over.
        estimator = RidgeCV(alphas=_RIDGE_ALPHA_GRID)
        resolved = _filtered_params(estimator, params, {})
        resolved.setdefault("alphas", _RIDGE_ALPHA_GRID)
        return _wrap_sklearn(RidgeCV(**resolved), scale=True)

    if model_type == "random_forest":
        estimator = RandomForestRegressor()
        resolved = _filtered_params(estimator, params, {"random_state": 42, "n_jobs": -1})
        return _wrap_sklearn(RandomForestRegressor(**resolved))

    if model_type == "extra_trees":
        estimator = ExtraTreesRegressor()
        resolved = _filtered_params(estimator, params, {"random_state": 42, "n_jobs": -1})
        return _wrap_sklearn(ExtraTreesRegressor(**resolved))

    if model_type == "gradient_boosting":
        estimator = GradientBoostingRegressor()
        resolved = _filtered_params(estimator, params, {"random_state": 42})
        return _wrap_sklearn(GradientBoostingRegressor(**resolved))

    supported = ", ".join(sorted(_MODEL_LABELS))
    raise ValueError(f"Unsupported model type '{model_type}'. Supported types: {supported}")


def extract_feature_importance(model: Any, feature_names: Iterable[str]) -> pd.DataFrame:
    """Extract a normalized feature importance table when the estimator exposes one."""
    features = list(feature_names)
    model = _final_estimator(model)

    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        importance = np.abs(np.asarray(model.coef_, dtype=float).reshape(-1))
    else:
        importance = np.zeros(len(features), dtype=float)

    if importance.shape[0] != len(features):
        raise ValueError("Feature importance length does not match feature names")

    return pd.DataFrame({"feature": features, "importance": importance}).sort_values(
        "importance",
        ascending=False,
    )
