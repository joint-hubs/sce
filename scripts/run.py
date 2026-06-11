"""
@module: scripts.run
@depends: sce, tomllib
@exports: run_experiment, run_all, generate_figures
@data_flow: config -> data -> SCE -> model -> metrics -> results

Main experiment runner for SCE validation.

Usage:
    python scripts/run.py --dataset rental_poland_short
    python scripts/run.py --all
    python scripts/run.py --generate-figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import tomllib
except ImportError:
    import tomli as tomllib

# Add parent to path for sce imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sce import (
    CleanupConfig,
    ContextConfig,
    StatisticalContextEngine,
    FeatureCombinationSearch,
    compute_lm_statistics,
    resolve_model_presets,
    aggregate_importance,
    run_iterative_pruning,
    SUPPORTED_CONTEXT_VARIANTS,
    get_context_variant_label,
    resolve_context_variant_methods,
)
from sce.config import AggregationMethod
from sce.model_presets import SUPPORTED_MODEL_TYPES
from sce.models import build_model, extract_feature_importance, get_model_label, model_supports_gpu

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data" / "parquet"
RESULTS_DIR = PROJECT_ROOT / "results"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(PROJECT_ROOT / 'experiment_debug.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


def _log_model_backend(model_type: str, use_gpu: bool = False) -> None:
    if use_gpu and not model_supports_gpu(model_type):
        logger.warning("GPU requested for %s, but this backend is CPU-only; falling back to CPU", model_type)

    if model_type == "xgboost":
        try:
            import xgboost as xgb
            logger.info(
                "Using XGBoost %s (gpu=%s)",
                getattr(xgb, "__version__", "unknown"),
                use_gpu,
            )
        except Exception as exc:
            logger.warning("XGBoost not available: %s", exc)
        return

    if model_type == "lightgbm":
        try:
            import lightgbm as lgb

            logger.info("Using LightGBM %s (gpu=%s)", getattr(lgb, "__version__", "unknown"), use_gpu)
        except Exception as exc:
            logger.warning("LightGBM not available: %s", exc)
        return

    if model_type == "catboost":
        try:
            import catboost

            logger.info("Using CatBoost %s (gpu=%s)", getattr(catboost, "__version__", "unknown"), use_gpu)
        except Exception as exc:
            logger.warning("CatBoost not available: %s", exc)
        return

    try:
        import sklearn
        logger.info(
            "Using model backend '%s' with scikit-learn %s (gpu=%s)",
            model_type,
            sklearn.__version__,
            use_gpu,
        )
    except Exception as exc:
        logger.warning("Unable to log scikit-learn version for %s: %s", model_type, exc)


def _log_sce_equations() -> None:
    logger.info("SCE Eq(1): phi_k(x_t) = S_k({y_s : s in N_k(t)})")
    logger.info("SCE Eq(2): Phi(x_t) = [phi^(1)(x_t), ..., phi^(K)(x_t)]")
    logger.info("SCE Eq(3): r_k,z = (y_t - mu_k) / (sigma_k + eps), r_k,ratio = y_t / (median_k + eps)")
    logger.info("SCE Eq(4): phi_cf^(k)(x_t) = S_k({y_s : s in N_k(t) \\ I_m})")


@dataclass
class ExperimentResult:
    """Container for experiment metrics."""
    dataset: str
    model_type: str
    context_variant: str
    categorical_mode: str
    baseline_rmse: float
    baseline_r2: float
    sce_rmse: float
    sce_r2: float
    rmse_improvement_pct: float
    r2_improvement_pct: float
    n_samples: int
    n_baseline_features: int
    n_sce_features: int
    runtime_seconds: float
    metadata: dict[str, Any] | None = None


@dataclass
class SCEEnrichmentResult:
    """Artifacts produced by SCE enrichment."""

    df_enriched: pd.DataFrame
    sce_feature_cols: list[str]
    all_sce_cols: list[str]


@dataclass
class DatasetSplit:
    """Train/test split for raw and optionally enriched data."""

    train_idx: pd.Index
    test_idx: pd.Index
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_enriched: pd.DataFrame | None = None
    test_enriched: pd.DataFrame | None = None


@dataclass
class PreparedFeatures:
    """Prepared feature artifacts with train-fitted preprocessing state."""

    X: pd.DataFrame
    y: pd.Series
    encoder: dict[str, pd.Index]
    droplist: list[tuple[str, str, float]]
    unseen_categorical: dict[str, dict[str, Any]] = field(default_factory=dict)


def _summarize_unseen_categories(series: pd.Series, categories: pd.Index) -> dict[str, Any]:
    """Summarize unseen category values against train-fitted categories."""
    if categories.empty:
        return {"unseen_count": 0, "unseen_rate": 0.0, "samples": []}

    unseen_mask = series.notna() & ~series.isin(categories)
    unseen_count = int(unseen_mask.sum())
    total_non_null = int(series.notna().sum())
    unseen_rate = float(unseen_count / total_non_null) if total_non_null else 0.0
    samples = []
    if unseen_count > 0:
        samples = [str(v) for v in pd.unique(series[unseen_mask])[:5]]

    return {
        "unseen_count": unseen_count,
        "unseen_rate": unseen_rate,
        "samples": samples,
    }


def _fit_categorical_encoder(
    train_df: pd.DataFrame,
    categorical_cols: list[str],
) -> dict[str, pd.Index]:
    """Fit categorical vocabularies on train data only."""
    encoder: dict[str, pd.Index] = {}
    for col in categorical_cols:
        categories = pd.Index(pd.Categorical(train_df[col]).categories)
        encoder[col] = categories
    return encoder


def _compute_pruning_droplist(
    df: pd.DataFrame,
    feature_cols: list[str],
    missing_threshold: float,
    drop_zero_variance: bool,
) -> list[tuple[str, str, float]]:
    """Compute columns to prune from the training partition only."""
    removed_cols: list[tuple[str, str, float]] = []
    for col in feature_cols:
        series = df[col]
        missing_rate = float(series.isna().mean())
        if missing_rate > missing_threshold:
            removed_cols.append((col, "missing_rate", missing_rate))
            continue
        if drop_zero_variance:
            nunique = series.nunique(dropna=True)
            if nunique <= 1:
                removed_cols.append((col, "zero_variance", float(nunique)))
    return removed_cols


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def _config_hash(config_path: Path) -> str:
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True)
        return bool(out.strip())
    except Exception:
        return True


def _build_run_id(config_name: str, git_sha: str, config_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{config_name}__{git_sha[:8]}__{config_hash[:8]}__{timestamp}"


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object per line, creating parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")


def _is_target_global_feature(feature: str, target_col: str) -> bool:
    """Heuristic check for target-derived global feature names."""
    if not feature:
        return False
    lowered = feature.lower()
    target = target_col.lower()
    has_target_pattern = lowered.startswith(f"{target}_") or f"_{target}_" in lowered
    has_global_pattern = lowered.startswith("global_") or "_global_" in lowered or lowered.endswith("_global")
    return has_target_pattern and has_global_pattern


def _load_latest_diagnostics(config_name: str) -> dict[str, Any]:
    """Load the most recent saved diagnostic payload of each type for a dataset.

    Diagnostics CLIs (scripts/diagnostics/*) persist their results under
    ``results/diagnostics/<dataset>/<diagnostic>_<timestamp>.json``. Report-grade
    promotion requires every diagnostic to be present.
    """
    diagnostics: dict[str, Any] = {
        "permuted_target": None,
        "shuffled_groups": None,
        "crossfit_ab": None,
        "feature_dominance": None,
    }
    base = PROJECT_ROOT / "results" / "diagnostics" / config_name
    if not base.exists():
        return diagnostics
    for diag_name in list(diagnostics):
        candidates = sorted(base.glob(f"{diag_name}_*.json"))
        if not candidates:
            continue
        try:
            diagnostics[diag_name] = json.loads(candidates[-1].read_text(encoding="utf-8"))
            diagnostics[diag_name]["source_file"] = str(candidates[-1])
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Could not load diagnostic %s: %s", candidates[-1], exc)
    return diagnostics


def _evaluate_promotion(
    run_grade: str,
    target_col: str,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate report-grade promotion gates and return promotion payload."""
    blocked_by: list[str] = []
    if run_grade == "report-grade":
        for diag_name in ["permuted_target", "shuffled_groups", "crossfit_ab", "feature_dominance"]:
            payload = diagnostics.get(diag_name)
            if payload is None:
                blocked_by.append(f"missing_diagnostic:{diag_name}")
            elif isinstance(payload, dict) and payload.get("pass") is False:
                blocked_by.append(f"diagnostic_failed:{diag_name}")

        dominance = diagnostics.get("feature_dominance")
        if isinstance(dominance, dict) and dominance.get("dominated"):
            top_features = dominance.get("top_features") or []
            lead_feature = str(top_features[0]) if top_features else ""
            if _is_target_global_feature(lead_feature, target_col):
                blocked_by.append("feature_dominance:target_global")

    promoted_to_report_grade = run_grade == "report-grade" and not blocked_by
    return {
        "promoted_to_report_grade": promoted_to_report_grade,
        "blocked_by": blocked_by,
    }


def _record_failure_case(
    run_id: str,
    config_name: str,
    model_type: str,
    rmse_improvement_pct: float,
    runtime_seconds: float,
    baseline_runtime_seconds: float | None,
    results_file: Path | None = None,
) -> None:
    """Persist low-improvement or high-runtime runs to a JSONL ledger."""
    reasons: list[str] = []
    if rmse_improvement_pct < 1.0:
        reasons.append("rmse_improvement_lt_1pct")
    if baseline_runtime_seconds and runtime_seconds > 2.0 * baseline_runtime_seconds:
        reasons.append("runtime_gt_2x_baseline")
    if not reasons:
        return

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "dataset": config_name,
        "model_type": model_type,
        "reason": reasons,
        "metrics": {
            "rmse_improvement_pct": float(rmse_improvement_pct),
            "runtime_seconds": float(runtime_seconds),
            "baseline_runtime_seconds": float(baseline_runtime_seconds) if baseline_runtime_seconds is not None else None,
        },
    }
    _append_jsonl(results_file or (RESULTS_DIR / "failure_cases.jsonl"), payload)


def _record_report_grade_block(
    run_id: str,
    config_name: str,
    blocked_by: list[str],
    config_path: str,
    results_file: Path | None = None,
) -> None:
    """Persist blocked report-grade attempts for auditability."""
    if not blocked_by:
        return
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_id": run_id,
        "dataset": config_name,
        "config_path": config_path,
        "blocked_by": blocked_by,
    }
    _append_jsonl(results_file or (RESULTS_DIR / "report_grade_blocks.jsonl"), payload)


def _enforce_run_grade_policy(run_grade: str, allow_dirty: bool = False) -> None:
    if run_grade != "report-grade":
        return
    if _git_dirty() and not allow_dirty:
        raise RuntimeError("report-grade runs require a clean git tree (or pass --allow-dirty)")


def _collect_run_metadata(
    config_name: str,
    config_path: Path,
    config: dict[str, Any],
    run_grade: str,
    split: DatasetSplit,
    sce_config: ContextConfig,
    model_type: str,
    model_params: dict[str, Any],
    runtime_seconds: float,
    metrics: dict[str, Any],
    n_rows_loaded: int,
    n_rows_after_filter: int,
    diagnostics: dict[str, Any] | None = None,
    report_grade_block_file: Path | None = None,
) -> dict[str, Any]:
    git_sha = _git_sha()
    git_dirty = _git_dirty()
    cfg_hash = _config_hash(config_path)
    run_id = _build_run_id(config_name, git_sha, cfg_hash)

    if run_grade == "report-grade" and git_sha == "unknown":
        raise RuntimeError("report-grade run requires a valid git SHA")

    split_cfg = config.get("split", {})
    sce_cfg = config.get("sce", {})
    dataset_cfg = config.get("dataset", {})
    feature_pruning = config.get("run", {}).get("feature_pruning", {})
    try:
        config_path_str = str(config_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        config_path_str = str(config_path)

    diagnostics_payload = diagnostics or {
        "permuted_target": None,
        "shuffled_groups": None,
        "crossfit_ab": None,
        "feature_dominance": None,
    }
    promotion = _evaluate_promotion(
        run_grade=run_grade,
        target_col=config["target"]["column"],
        diagnostics=diagnostics_payload,
    )

    if run_grade == "report-grade" and promotion["blocked_by"]:
        _record_report_grade_block(
            run_id=run_id,
            config_name=config_name,
            blocked_by=promotion["blocked_by"],
            config_path=config_path_str,
            results_file=report_grade_block_file,
        )
        blocked = ", ".join(promotion["blocked_by"])
        logger.error("report-grade blocked for %s: %s", config_name, blocked)
        raise RuntimeError(f"report-grade promotion blocked: {blocked}")

    return {
        "run_id": run_id,
        "run_grade": run_grade,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "config_path": config_path_str,
        "config_hash": cfg_hash,
        "dataset": {
            "name": config_name,
            "source": dataset_cfg.get("source", "local"),
            "path": dataset_cfg.get("path", ""),
            "n_rows_loaded": int(n_rows_loaded),
            "n_rows_after_filter": int(n_rows_after_filter),
            "target_col": config["target"]["column"],
        },
        "split": {
            "strategy": split_cfg.get("strategy", "random"),
            "test_size": split_cfg.get("test_size", 0.2),
            "test_periods": split_cfg.get("test_periods"),
            "time_col": split_cfg.get("time_col"),
            "seed": split_cfg.get("random_state", 42),
            "n_train": int(len(split.train_idx)),
            "n_test": int(len(split.test_idx)),
        },
        "sce": {
            "context_variant": metrics.get("context_variant", "sce"),
            "categorical_mode": metrics.get("categorical_mode", "manual"),
            "categorical_cols_resolved": sce_config.categorical_cols,
            "aggregations": [m.value for m in sce_config.aggregations],
            "min_group_size": sce_config.min_group_size,
            "use_cross_fitting": sce_config.use_cross_fitting,
            "cross_fit_strategy": sce_config.cross_fit_strategy,
            "n_folds": sce_config.n_folds,
            "rolling_max_train_size": sce_config.rolling_max_train_size,
            "rolling_test_size": sce_config.rolling_test_size,
            "rolling_gap": sce_config.rolling_gap,
            "include_global_stats": sce_config.include_global_stats,
            "include_interactions": sce_config.include_interactions,
            "max_interaction_depth": sce_config.max_interaction_depth,
            "include_fold_variance": sce_config.include_fold_variance,
            "fold_variance_features": sce_config.fold_variance_features,
            "include_relative_features": sce_config.include_relative_features,
            "n_sce_features": int(metrics.get("n_sce_features", 0)),
            "n_context_features": int(metrics.get("n_sce_features", 0)),
        },
        "model": {
            "type": model_type,
            "preset": metrics.get("model_preset", "default"),
            "params": model_params,
        },
        "metrics": {
            "baseline_rmse": float(metrics.get("baseline_rmse", 0.0)),
            "baseline_r2": float(metrics.get("baseline_r2", 0.0)),
            "sce_rmse": float(metrics.get("sce_rmse", 0.0)),
            "sce_r2": float(metrics.get("sce_r2", 0.0)),
            "rmse_improvement_pct": float(metrics.get("rmse_improvement_pct", 0.0)),
            "r2_improvement_pp": float(metrics.get("r2_improvement_pp", 0.0)),
            "runtime_seconds": float(runtime_seconds),
            "n_baseline_features": int(metrics.get("n_baseline_features", 0)),
            "n_sce_features": int(metrics.get("n_sce_features", 0)),
        },
        "diagnostics": diagnostics_payload,
        "promotion": promotion,
        "feature_pruning": {
            "missing_threshold": feature_pruning.get("missing_threshold", 0.2),
            "drop_zero_variance": feature_pruning.get("drop_zero_variance", True),
        },
    }


def audit_feature_dominance(
    importance_csv: Path,
    top_k: int = 3,
    threshold_pct: float = 70.0,
) -> dict[str, Any]:
    """Audit whether feature importance is dominated by top-k features."""
    if not importance_csv.exists():
        return {"top_k_share_pct": 0.0, "top_features": [], "dominated": False}

    df = pd.read_csv(importance_csv)
    if df.empty:
        return {"top_k_share_pct": 0.0, "top_features": [], "dominated": False}

    candidate_cols = [c for c in ["avg_importance", "importance", "mean_importance"] if c in df.columns]
    if not candidate_cols:
        return {"top_k_share_pct": 0.0, "top_features": [], "dominated": False}

    score_col = candidate_cols[0]
    sorted_df = df.sort_values(score_col, ascending=False)
    total = float(sorted_df[score_col].sum())
    if total <= 0:
        return {"top_k_share_pct": 0.0, "top_features": [], "dominated": False}

    top_df = sorted_df.head(top_k)
    share = float(top_df[score_col].sum() / total * 100)
    top_features = top_df["feature"].astype(str).tolist() if "feature" in top_df.columns else []

    return {
        "top_k": int(top_k),
        "threshold_pct": float(threshold_pct),
        "top_k_share_pct": share,
        "top_features": top_features,
        "dominated": share > threshold_pct,
    }


def load_config(config_path: Path) -> dict[str, Any]:
    """Load TOML configuration file."""
    with config_path.open("rb") as f:
        return tomllib.load(f)


def _resolve_sample_size(config_name: str, sample_size: int | None) -> int | None:
    """Resolve dataset sample size, using repo defaults when not specified."""
    if sample_size is not None:
        return sample_size

    if "uae" in config_name.lower() or "dubai" in config_name.lower():
        return 50000

    return None


def _filter_target_rows(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Drop rows with missing or non-positive targets and log the effect."""
    initial_rows = len(df)
    missing_target = df[target_col].isna().sum()
    zero_or_neg = (df[target_col] <= 0).sum()

    logger.debug("Missing target values: %s", missing_target)
    logger.debug("Zero or negative target values: %s", zero_or_neg)

    filtered = df.dropna(subset=[target_col])
    filtered = filtered[filtered[target_col] > 0]

    rows_dropped = initial_rows - len(filtered)
    if rows_dropped > 0:
        logger.warning(
            "Dropped %s rows (%.1f%%) due to missing or invalid target values",
            rows_dropped,
            rows_dropped / initial_rows * 100,
        )

    return filtered


def _sample_dataset(df: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    """Sample dataset deterministically when a cap is provided."""
    if sample_size and len(df) > sample_size:
        sampled = df.sample(n=sample_size, random_state=42)
        logger.info("Sampled dataset to %s rows", sample_size)
        return sampled
    return df


def _build_sce_config(
    config: dict[str, Any],
    target_col: str,
    cleanup: bool = False,
    context_variant: str = "sce",
    categorical_mode: str = "manual",
) -> ContextConfig:
    """Build ContextConfig from experiment config."""
    sce_cfg = config.get("sce", {})
    split_cfg = config.get("split", {})
    split_strategy = split_cfg.get("strategy", "random")
    sce_use_cf = sce_cfg.get("use_cross_fitting", True)
    sce_cf_strategy = sce_cfg.get(
        "cross_fit_strategy",
        "rolling" if split_strategy == "temporal" else "random",
    )

    if split_strategy == "temporal" and sce_use_cf and sce_cf_strategy == "random":
        raise ValueError(
            "Temporal split forbids random cross-fit (causes temporal leakage). "
            "Set sce.cross_fit_strategy='rolling' (recommended) or sce.use_cross_fitting=false."
        )

    agg_names = sce_cfg.get("aggregations", ["mean", "std", "median", "count"])
    base_aggregations = [AggregationMethod(name) for name in agg_names]
    aggregations = resolve_context_variant_methods(context_variant, base_aggregations)
    explicit_sce_categoricals = sce_cfg.get("categorical_cols")
    feature_categoricals = config.get("features", {}).get("categorical", [])
    if categorical_mode == "auto":
        resolved_categoricals = None
    else:
        resolved_categoricals = explicit_sce_categoricals
        if resolved_categoricals is None:
            resolved_categoricals = feature_categoricals or None
    cleanup_config = CleanupConfig() if cleanup else None
    include_fold_variance = sce_cfg.get("include_fold_variance", True) if context_variant == "sce" else False
    add_backoff_depth = sce_cfg.get("add_backoff_depth", False) if context_variant == "sce" else False

    logger.info(
        "Context config (%s): aggregations=%s, cross_fitting=%s, folds=%s, fold_variance=%s, variance_features=%s, backoff_depth=%s",
        context_variant,
        [method.value for method in aggregations],
        sce_cfg.get("use_cross_fitting", True),
        sce_cfg.get("n_folds", 5),
        include_fold_variance,
        sce_cfg.get("fold_variance_features", ["std", "lower", "upper"]),
        add_backoff_depth,
    )
    logger.info(
        "Categorical grouping mode: %s (%s)",
        categorical_mode,
        "auto-detect from dataframe" if resolved_categoricals is None else resolved_categoricals,
    )

    return ContextConfig(
        target_col=target_col,
        categorical_cols=resolved_categoricals,
        min_categorical_columns=sce_cfg.get("min_categorical_columns", 1),
        aggregations=aggregations,
        min_group_size=sce_cfg.get("min_group_size", 3),
        use_cross_fitting=sce_cfg.get("use_cross_fitting", True),
        cross_fit_strategy=sce_cf_strategy,
        time_col=split_cfg.get("time_col"),
        n_folds=sce_cfg.get("n_folds", 5),
        random_state=sce_cfg.get("random_state", split_cfg.get("random_state", 42)),
        rolling_max_train_size=sce_cfg.get("rolling_max_train_size"),
        rolling_test_size=sce_cfg.get("rolling_test_size"),
        rolling_gap=sce_cfg.get("rolling_gap", 0),
        include_interactions=sce_cfg.get("include_interactions", True),
        max_interaction_depth=sce_cfg.get("max_interaction_depth", 2),
        max_cardinality=sce_cfg.get("max_cardinality", 100),
        include_fold_variance=include_fold_variance,
        fold_variance_features=sce_cfg.get("fold_variance_features", ["std", "lower", "upper"]),
        add_backoff_depth=add_backoff_depth,
        cleanup_config=cleanup_config,
    )


def _resolve_context_variant(
    config: dict[str, Any],
    context_variant_override: str | None = None,
) -> str:
    run_cfg = config.get("run", {}) if isinstance(config, dict) else {}
    return (context_variant_override or run_cfg.get("context_variant") or "sce").strip().lower()


def _resolve_categorical_mode(
    config: dict[str, Any],
    categorical_mode_override: str | None = None,
) -> str:
    run_cfg = config.get("run", {}) if isinstance(config, dict) else {}
    mode = (categorical_mode_override or run_cfg.get("categorical_mode") or "manual").strip().lower()
    if mode not in {"manual", "auto"}:
        raise ValueError("categorical_mode must be one of: manual, auto")
    return mode


def _log_cleanup_summary(engine: StatisticalContextEngine) -> None:
    """Log cleanup summary when cleanup pipeline is enabled."""
    if not getattr(engine, "_cleanup_report", None):
        return

    report = engine._cleanup_report
    logger.info(
        "Cleanup summary: original=%s final=%s removed=%s",
        report.original_features,
        report.final_features,
        report.total_removed,
    )
    logger.info(
        "Cleanup breakdown: constant=%s leakage=%s correlation=%s hierarchy=%s vif=%s",
        len(report.constant_removed),
        len(report.leakage_removed),
        len(report.correlation_removed),
        len(report.hierarchy_removed),
        len(report.vif_removed),
    )


def _log_backoff_stats(df_enriched: pd.DataFrame) -> None:
    """Log aggregate backoff-depth statistics when present."""
    backoff_cols = [c for c in df_enriched.columns if c.endswith("_backoff_depth")]
    if not backoff_cols:
        return

    backoff_values = df_enriched[backoff_cols]
    mean_depth = float(backoff_values.mean().mean())
    max_depth = int(backoff_values.max().max())
    pct_backoff = float((backoff_values > 0).mean().mean())
    logger.info(
        "Backoff depth stats: mean=%.3f max=%s pct_backoff=%.2f",
        mean_depth,
        max_depth,
        pct_backoff,
    )


def _run_sce_enrichment(
    df: pd.DataFrame,
    config: dict[str, Any],
    target_col: str,
    cleanup: bool = False,
    transform_df: pd.DataFrame | None = None,
    context_variant: str = "sce",
    categorical_mode: str = "manual",
) -> SCEEnrichmentResult:
    """Run SCE enrichment and collect generated feature metadata.

    When ``transform_df`` is provided, the engine is fit on ``df`` only and then
    applied to ``transform_df`` using train-derived statistics. The returned frame
    preserves the original indices of both partitions so downstream code can align
    them back to the raw split.
    """
    _log_sce_equations()
    sce_config = _build_sce_config(
        config,
        target_col,
        cleanup=cleanup,
        context_variant=context_variant,
        categorical_mode=categorical_mode,
    )
    engine = StatisticalContextEngine(sce_config)

    if transform_df is None:
        df_enriched = engine.fit_transform(df)
    else:
        logger.info(
            "Evaluation protocol: split-first train-fit/test-transform using train-derived SCE statistics"
        )
        train_enriched = engine.fit_transform(df)
        test_features = transform_df.drop(columns=[target_col])
        test_enriched = engine.transform(test_features)
        test_enriched[target_col] = transform_df[target_col]
        df_enriched = pd.concat([train_enriched, test_enriched], axis=0).sort_index()

    _log_cleanup_summary(engine)
    _log_backoff_stats(df_enriched)

    sce_feature_cols = [c for c in df_enriched.columns if c not in df.columns]
    logger.info("Generated %s SCE features", len(sce_feature_cols))

    # Target-derived ratio features remain disabled in evaluation.
    all_sce_cols = list(sce_feature_cols)
    logger.info(
        "Total enriched features: %s (SCE: %s, Ratios: 0)",
        len(all_sce_cols),
        len(sce_feature_cols),
    )

    return SCEEnrichmentResult(
        df_enriched=df_enriched,
        sce_feature_cols=sce_feature_cols,
        all_sce_cols=all_sce_cols,
    )


def _split_dataset(
    df: pd.DataFrame,
    target_col: str,
    config: dict[str, Any] | None = None,
    df_enriched: pd.DataFrame | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Create a deterministic train/test split for raw and enriched data."""
    split_cfg = (config or {}).get("split", {})
    strategy = split_cfg.get("strategy", "random")

    if strategy == "temporal":
        time_col = split_cfg.get("time_col")
        if not time_col or time_col not in df.columns:
            raise ValueError("Temporal split requires split.time_col to exist in the dataset")

        sorted_df = df.sort_values(time_col, kind="mergesort")
        unique_times = pd.Index(pd.unique(sorted_df[time_col]))
        test_periods = split_cfg.get("test_periods")
        if test_periods is None:
            test_fraction = float(split_cfg.get("test_size", test_size))
            test_periods = max(1, int(np.ceil(len(unique_times) * test_fraction)))

        if int(test_periods) >= len(unique_times):
            raise ValueError("Temporal split requires fewer test periods than total unique periods")

        test_times = set(unique_times[-int(test_periods) :].tolist())
        test_mask = sorted_df[time_col].isin(test_times)
        train_idx = sorted_df.loc[~test_mask].index
        test_idx = sorted_df.loc[test_mask].index

        logger.info(
            "Temporal split on '%s': train_periods=%s test_periods=%s train_end=%s test_start=%s",
            time_col,
            len(unique_times) - int(test_periods),
            int(test_periods),
            unique_times[-int(test_periods) - 1],
            unique_times[-int(test_periods)],
        )
    else:
        train_idx, test_idx = train_test_split(
            df.index,
            test_size=split_cfg.get("test_size", test_size),
            random_state=split_cfg.get("random_state", random_state),
        )

    split = DatasetSplit(
        train_idx=pd.Index(train_idx),
        test_idx=pd.Index(test_idx),
        train_df=df.loc[train_idx],
        test_df=df.loc[test_idx],
        train_enriched=df_enriched.loc[train_idx] if df_enriched is not None else None,
        test_enriched=df_enriched.loc[test_idx] if df_enriched is not None else None,
    )

    logger.info("Train size: %s (%.1f%%)", len(split.train_idx), len(split.train_idx) / len(df) * 100)
    logger.info("Test size: %s (%.1f%%)", len(split.test_idx), len(split.test_idx) / len(df) * 100)

    train_target = split.train_df[target_col]
    test_target = split.test_df[target_col]
    logger.info(
        "Target distribution train/test summary: %s",
        json.dumps(
            {
                "train": {
                    "min": float(train_target.min()),
                    "max": float(train_target.max()),
                    "mean": float(train_target.mean()),
                    "std": float(train_target.std()),
                },
                "test": {
                    "min": float(test_target.min()),
                    "max": float(test_target.max()),
                    "mean": float(test_target.mean()),
                    "std": float(test_target.std()),
                },
                "similarity": {
                    "mean_abs_diff": abs(float(train_target.mean()) - float(test_target.mean())),
                    "std_abs_diff": abs(float(train_target.std()) - float(test_target.std())),
                },
            }
        ),
    )

    return split


def _align_common_columns(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Align train/test matrices while preserving train-column order."""
    common_cols = [col for col in X_train.columns if col in X_test.columns]
    return X_train[common_cols], X_test[common_cols], common_cols


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load dataset based on config. Downloads remote files on demand if needed."""
    data_path = PROJECT_ROOT / config["dataset"]["path"]
    logger.info(f"Loading dataset from: {data_path}")

    # Check if dataset needs to be downloaded
    if not data_path.exists():
        source = config["dataset"].get("source", "local")
        if source == "remote":
            logger.info("Dataset not found locally, downloading from configured remote source...")
            # Import here to avoid dependency issues
            import subprocess

            download_script = PROJECT_ROOT / "scripts" / "download_datasets.py"
            dataset_name = data_path.name
            result = subprocess.run(
                [sys.executable, str(download_script), "--dataset", dataset_name],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                logger.error(f"Download failed: {result.stderr}")
                raise FileNotFoundError(f"Failed to download dataset: {dataset_name}")
            logger.info("Download complete")
        else:
            logger.error(f"Dataset not found: {data_path}")
            raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
    logger.debug(f"Columns: {list(df.columns)}")
    logger.debug(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    return df


def prepare_features(
    df: pd.DataFrame,
    config: dict[str, Any],
    target_col: str,
    encoder: dict[str, pd.Index] | None = None,
    droplist: list[tuple[str, str, float]] | None = None,
) -> PreparedFeatures:
    """Prepare feature matrix and target vector with train-only fitted preprocessing."""
    logger.debug(f"Preparing features from {len(df)} rows")

    # Get feature columns from config
    numeric_cols = config.get("features", {}).get("numeric", [])
    categorical_cols = config.get("features", {}).get("categorical", [])

    logger.debug(f"Config numeric columns: {numeric_cols}")
    logger.debug(f"Config categorical columns: {categorical_cols}")

    # Filter to existing columns
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    logger.info(f"Using {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features")
    logger.debug(f"Numeric features: {numeric_cols}")
    logger.debug(f"Categorical features: {categorical_cols}")

    # Build feature dataframe
    feature_cols = numeric_cols + categorical_cols
    working_df = df[feature_cols + [target_col]].copy()

    # Coerce numerics (no imputation)
    for col in numeric_cols:
        if col in working_df.columns:
            working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

    # Drop columns with too much missingness or zero variance
    run_cfg = config.get("run", {})
    pruning_cfg = run_cfg.get("feature_pruning", {})
    missing_threshold = pruning_cfg.get("missing_threshold", 0.2)
    drop_zero_variance = pruning_cfg.get("drop_zero_variance", True)

    resolved_droplist = droplist
    if resolved_droplist is None:
        resolved_droplist = _compute_pruning_droplist(
            working_df,
            feature_cols,
            missing_threshold,
            drop_zero_variance,
        )

    if resolved_droplist:
        removed_set = {col for col, _, _ in resolved_droplist}
        numeric_cols = [c for c in numeric_cols if c not in removed_set]
        categorical_cols = [c for c in categorical_cols if c not in removed_set]
        feature_cols = numeric_cols + categorical_cols
        working_df = working_df[feature_cols + [target_col]].copy()

        logger.warning(
            "Pruned %s feature columns (missing_threshold=%.2f, drop_zero_variance=%s)",
            len(resolved_droplist),
            missing_threshold,
            drop_zero_variance,
        )
        for col, reason, value in resolved_droplist:
            if reason == "missing_rate":
                logger.warning("- Dropped '%s' due to missing_rate=%.2f", col, value)
            else:
                logger.warning("- Dropped '%s' due to zero_variance (nunique=%s)", col, int(value))

    if not feature_cols:
        raise ValueError("No usable features remain after pruning.")

    # Drop rows with missing values in any feature
    before_rows = len(working_df)
    working_df = working_df.dropna(subset=feature_cols)
    dropped = before_rows - len(working_df)
    if dropped > 0:
        logger.warning(
            "Dropped %s rows due to missing feature values (no imputation)",
            dropped,
        )

    X = working_df[feature_cols].copy()

    # Encode categoricals with train-fitted categories only.
    unseen_categorical: dict[str, dict[str, Any]] = {}
    resolved_encoder = encoder
    if resolved_encoder is None:
        resolved_encoder = _fit_categorical_encoder(working_df, categorical_cols)

    for col in categorical_cols:
        categories = resolved_encoder.get(col, pd.Index([]))
        if categories.empty:
            X[col] = pd.Categorical(X[col]).codes
            continue
        # Keep unseen values as NaN for tree models (e.g., XGBoost) instead of synthetic "unknown" rows.
        unseen_summary = _summarize_unseen_categories(X[col], categories)
        unseen_categorical[col] = unseen_summary
        if unseen_summary["unseen_count"] > 0:
            logger.warning(
                "Column '%s': mapped %s unseen categories to NaN (rate=%.4f, samples=%s)",
                col,
                unseen_summary["unseen_count"],
                unseen_summary["unseen_rate"],
                unseen_summary["samples"],
            )
        encoded = pd.Categorical(X[col], categories=categories).codes.astype(float)
        encoded[encoded == -1] = np.nan
        X[col] = encoded

    y = working_df[target_col].copy()
    logger.info(f"Target '{target_col}': min={y.min():.2f}, max={y.max():.2f}, "
                f"mean={y.mean():.2f}, median={y.median():.2f}, std={y.std():.2f}")

    return PreparedFeatures(
        X=X,
        y=y,
        encoder=resolved_encoder,
        droplist=resolved_droplist,
        unseen_categorical=unseen_categorical,
    )


def create_ratio_features(df: pd.DataFrame, sce_cols: list[str], target_col: str) -> pd.DataFrame:
    """
    Create ratio features (Eq. 3 from paper).
    
    These relative features capture how each observation compares to its group,
    which is crucial for model performance.
    """
    logger.debug(f"Creating ratio features from {len(sce_cols)} SCE columns")
    df = df.copy()
    target_vals = df[target_col].values
    
    ratio_count = 0
    for col in sce_cols:
        if '_mean' in col or '_median' in col:
            # Create ratio: value / group_statistic
            group_vals = df[col].values
            # Avoid division by zero
            safe_vals = np.where(group_vals != 0, group_vals, 1.0)
            ratio_col = col.replace('_mean', '_ratio').replace('_median', '_ratio_med')
            df[ratio_col] = target_vals / safe_vals
            # Clip extreme values
            df[ratio_col] = df[ratio_col].clip(-10, 10)
            ratio_count += 1
            
            logger.debug(f"Created ratio feature '{ratio_col}': "
                        f"min={df[ratio_col].min():.3f}, max={df[ratio_col].max():.3f}, "
                        f"mean={df[ratio_col].mean():.3f}")
    
    logger.info(f"Created {ratio_count} ratio features")
    return df


def _resolve_standard_model(
    config: dict[str, Any],
    model_type_override: str | None = None,
    use_gpu_override: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    model_cfg = dict(config.get("model", {}))
    if model_type_override:
        model_cfg["type"] = model_type_override
    if use_gpu_override is not None:
        model_cfg["use_gpu"] = use_gpu_override

    model_type, _, presets = resolve_model_presets({}, model_cfg)
    resolved_params = dict(presets["default"])
    resolved_params.update({k: v for k, v in model_cfg.items() if k != "type"})
    return model_type, resolved_params


def train_configured_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    model_params: dict[str, Any],
):
    """Train the configured downstream model."""
    model = build_model(model_type, model_params)
    return model.fit(X_train, y_train)


def _create_output_dir(config_name: str, suffix: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = RESULTS_DIR / f"{config_name}_{suffix}_{timestamp}"
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    return output_dir


def run_search_experiment(
    config_name: str,
    sample_size: int | None = None,
    sampling_pct: float | None = None,
    p_threshold: float | None = None,
    run_report: bool = False,
    model_presets_override: list[str] | None = None,
    model_type_override: str | None = None,
    use_gpu_override: bool | None = None,
    context_variant_override: str | None = None,
    categorical_mode_override: str | None = None,
    cleanup: bool = False,
    run_grade: str = "exploratory",
    allow_dirty: bool = False,
) -> Path:
    """Run combinatorial search + reporting pipeline."""
    _enforce_run_grade_policy(run_grade, allow_dirty)

    search_start = time.time()
    config_path = CONFIGS_DIR / f"{config_name}.toml"
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        raise FileNotFoundError(f"Config not found: {config_path}")

    logger.info("=" * 80)
    logger.info("STARTING SEARCH EXPERIMENT: %s", config_name)
    logger.info("=" * 80)

    config = load_config(config_path)
    run_cfg = config.get("run", {})
    search_cfg = run_cfg.get("search", {})
    lm_cfg = run_cfg.get("lm_selection", {})
    model_cfg = dict(config.get("model", {}))
    if model_type_override:
        model_cfg["type"] = model_type_override
    if use_gpu_override is not None:
        model_cfg["use_gpu"] = use_gpu_override
    model_type = str(model_cfg.get("type", "xgboost")).lower()
    use_gpu = bool(model_cfg.get("use_gpu", False))
    context_variant = _resolve_context_variant(config, context_variant_override)
    categorical_mode = _resolve_categorical_mode(config, categorical_mode_override)

    _log_model_backend(model_type, use_gpu=use_gpu)

    df = load_dataset(config)
    n_rows_loaded = len(df)
    target_col = config["target"]["column"]

    df = _filter_target_rows(df, target_col)
    df = _sample_dataset(df, _resolve_sample_size(config_name, sample_size))
    n_rows_after_filter = len(df)

    output_dir = _create_output_dir(config_name, "search")

    # Clean predictive evaluation: split first, fit SCE on train only, transform test.
    split = _split_dataset(df, target_col, config=config)
    enrichment = _run_sce_enrichment(
        split.train_df,
        config,
        target_col,
        cleanup=cleanup,
        transform_df=split.test_df,
        context_variant=context_variant,
        categorical_mode=categorical_mode,
    )
    df_enriched = enrichment.df_enriched
    sce_feature_cols = enrichment.sce_feature_cols

    train_df = split.train_df
    test_df = split.test_df
    train_enriched = df_enriched.loc[split.train_idx]
    test_enriched = df_enriched.loc[split.test_idx]

    train_base = prepare_features(train_df, config, target_col)
    test_base = prepare_features(
        test_df,
        config,
        target_col,
        encoder=train_base.encoder,
        droplist=train_base.droplist,
    )
    X_train_base, y_train = train_base.X, train_base.y
    X_test_base, y_test = test_base.X, test_base.y

    # Add SCE features - align on index from prepare_features (which dropped rows)
    sce_train_df = train_enriched.loc[X_train_base.index, sce_feature_cols]
    sce_test_df = test_enriched.loc[X_test_base.index, sce_feature_cols]

    X_train_all = pd.concat([X_train_base, sce_train_df], axis=1)
    X_test_all = pd.concat([X_test_base, sce_test_df], axis=1)
    X_train_all = X_train_all.replace([np.inf, -np.inf], np.nan)
    X_test_all = X_test_all.replace([np.inf, -np.inf], np.nan)

    train_before = len(X_train_all)
    test_before = len(X_test_all)
    min_non_missing_pct = search_cfg.get("min_non_missing_pct", run_cfg.get("min_non_missing_pct", 0.5))
    min_non_missing = max(1, int(min_non_missing_pct * X_train_all.shape[1]))
    X_train_all = X_train_all.dropna(thresh=min_non_missing)
    X_test_all = X_test_all.dropna(thresh=min_non_missing)
    if len(X_train_all) < train_before or len(X_test_all) < test_before:
        logger.warning(
            "Dropped %s train and %s test rows after SCE enrichment due to missing values "
            "(min_non_missing=%s of %s features)",
            train_before - len(X_train_all),
            test_before - len(X_test_all),
            min_non_missing,
            X_train_all.shape[1],
        )

    # Align base and enriched sets
    common_train_idx = X_train_base.index.intersection(X_train_all.index)
    common_test_idx = X_test_base.index.intersection(X_test_all.index)
    X_train_base = X_train_base.loc[common_train_idx]
    y_train = y_train.loc[common_train_idx]
    X_test_base = X_test_base.loc[common_test_idx]
    y_test = y_test.loc[common_test_idx]
    X_train_all = X_train_all.loc[common_train_idx]
    X_test_all = X_test_all.loc[common_test_idx]

    base_features = list(X_train_base.columns)
    context_features = sce_feature_cols

    # Resolve presets
    if model_presets_override:
        run_cfg = dict(run_cfg)
        preset_key = "xgboost_configs" if model_type == "xgboost" else f"{model_type}_configs"
        run_cfg[preset_key] = model_presets_override
    model_type, preset_names, preset_params = resolve_model_presets(run_cfg, model_cfg)

    baseline_runtime_seconds: float | None = None
    baseline_config_name = "default" if "default" in preset_params else (preset_names[0] if preset_names else None)
    if baseline_config_name:
        try:
            baseline_timing_start = time.perf_counter()
            X_train_base_timing = X_train_base.fillna(0)
            X_test_base_timing = X_test_base.fillna(0)
            baseline_model = train_configured_model(
                X_train_base_timing,
                y_train,
                model_type,
                preset_params[baseline_config_name],
            )
            _ = baseline_model.predict(X_test_base_timing)
            baseline_runtime_seconds = time.perf_counter() - baseline_timing_start
        except Exception as exc:
            logger.warning("Could not measure same-run baseline runtime for search: %s", exc)

    # LM statistics
    lm_enabled = lm_cfg.get("enabled", True)
    lm_threshold = p_threshold if p_threshold is not None else lm_cfg.get("p_threshold", run_cfg.get("p_threshold", 0.05))
    sig_base: list[str] = []
    sig_context: list[str] = []

    if lm_enabled:
        lm_base = compute_lm_statistics(X_train_all, y_train, base_features)
        lm_context = compute_lm_statistics(X_train_all, y_train, context_features)
        lm_base.feature_stats.to_csv(output_dir / "data" / "lm_base_statistics.csv", index=False)
        lm_context.feature_stats.to_csv(output_dir / "data" / "lm_context_statistics.csv", index=False)

        sig_base = lm_base.feature_stats[lm_base.feature_stats["p_value"] < lm_threshold]["feature"].tolist()
        sig_context = lm_context.feature_stats[lm_context.feature_stats["p_value"] < lm_threshold]["feature"].tolist()

    # Search configuration
    sampling_pct_val = sampling_pct if sampling_pct is not None else search_cfg.get("sampling_pct", 5.0)
    min_samples = search_cfg.get("min_configs", 50)
    max_samples = search_cfg.get("max_configs", 500)
    run_ablation = search_cfg.get("run_ablation", True)
    run_significance = search_cfg.get("run_significance_selection", True) and lm_enabled

    split_strategy = config.get("split", {}).get("strategy", "random")
    search_val_strategy = "tail" if split_strategy == "temporal" else "random"
    search_val_fraction = float(search_cfg.get("val_fraction", 0.2))
    logger.info(
        "Search selection protocol: internal %s validation split (%.0f%% of train); "
        "test set evaluated once for selected winners only",
        search_val_strategy,
        search_val_fraction * 100,
    )

    searcher = FeatureCombinationSearch(
        base_features=base_features,
        context_features=context_features,
        sampling_pct=sampling_pct_val,
        min_samples=min_samples,
        max_samples=max_samples,
        model_configs=preset_names,
        model_params=preset_params,
        model_type=model_type,
        run_ablation=run_ablation,
        run_significance_selection=run_significance,
        p_threshold=lm_threshold,
        val_fraction=search_val_fraction,
        val_strategy=search_val_strategy,
    )

    summary = searcher.search(X_train_all, y_train, X_test_all, y_test)

    # Baseline and best-SCE metrics are unbiased test-set scores: winners were
    # selected on the internal validation split and refit on the full train.
    baseline_rmse = float(summary.baseline_result.rmse)
    baseline_r2 = float(summary.baseline_result.r2)

    best_sce_rmse = float(summary.best_by_rmse.rmse)
    rmse_improvement_pct = 0.0
    if baseline_rmse > 0:
        rmse_improvement_pct = ((baseline_rmse - best_sce_rmse) / baseline_rmse) * 100.0

    final_results = [summary.baseline_result, summary.best_by_rmse, summary.best_by_r2]
    results_df = pd.DataFrame([
        {
            "config_id": r.config_id,
            "strategy": r.strategy,
            "model_config": r.model_config,
            "n_features": r.n_features,
            "n_base": r.n_base,
            "n_context": r.n_context,
            "rmse": r.rmse,
            "r2": r.r2,
            "mae": r.mae,
            "eval_set": r.eval_set,
            "val_rmse": r.val_rmse,
            "features": "|".join(r.features),
        }
        for r in list(summary.all_results) + final_results
    ])
    results_df.to_csv(output_dir / "data" / "model_comparison.csv", index=False)

    # Importance aggregation
    agg_importance = aggregate_importance(summary.all_results)
    if not agg_importance.empty:
        agg_importance.to_csv(output_dir / "data" / "aggregated_feature_importance.csv", index=False)

    # Pruning steps (default preset) — scored on the internal validation
    # split so the pruning trace does not consume the test set.
    X_prune_fit = X_train_all.iloc[searcher.fit_indices_]
    y_prune_fit = y_train.iloc[searcher.fit_indices_]
    X_prune_val = X_train_all.iloc[searcher.val_indices_]
    y_prune_val = y_train.iloc[searcher.val_indices_]
    pruning_results, removed_df = run_iterative_pruning(
        X_prune_fit,
        y_prune_fit,
        X_prune_val,
        y_prune_val,
        features=base_features + context_features,
        model_type=model_type,
        model_config_name="default",
        model_params=preset_params,
    )
    pd.DataFrame([r.__dict__ for r in pruning_results]).to_csv(
        output_dir / "data" / "pruning_trace.csv",
        index=False,
    )
    pd.DataFrame([r.__dict__ for r in pruning_results]).to_csv(
        output_dir / "data" / f"{model_type}_pruning_trace.csv",
        index=False,
    )
    removed_df.to_csv(output_dir / "data" / "pruning_removed_features.csv", index=False)
    removed_df.to_csv(output_dir / "data" / f"{model_type}_pruning_removed_features.csv", index=False)
    if model_type == "xgboost":
        pd.DataFrame([r.__dict__ for r in pruning_results]).to_csv(
            output_dir / "data" / "xgb_pruning_trace.csv",
            index=False,
        )
        removed_df.to_csv(output_dir / "data" / "xgb_pruning_removed_features.csv", index=False)

    # Metadata summary
    importance_path = output_dir / "data" / "aggregated_feature_importance.csv"
    feature_dominance = audit_feature_dominance(importance_path)
    if feature_dominance.get("dominated"):
        logger.warning(
            "Feature dominance warning: top-%s share %.2f%% exceeds %.2f%% (top_features=%s)",
            feature_dominance.get("top_k", 3),
            feature_dominance.get("top_k_share_pct", 0.0),
            feature_dominance.get("threshold_pct", 70.0),
            feature_dominance.get("top_features", []),
        )

    sce_config = _build_sce_config(
        config,
        target_col,
        cleanup=cleanup,
        context_variant=context_variant,
        categorical_mode=categorical_mode,
    )

    runtime_seconds = time.time() - search_start
    metadata = _collect_run_metadata(
        config_name=config_name,
        config_path=config_path,
        config=config,
        run_grade=run_grade,
        split=split,
        sce_config=sce_config,
        model_type=model_type,
        model_params=preset_params,
        runtime_seconds=runtime_seconds,
        metrics={
            "context_variant": context_variant,
            "categorical_mode": categorical_mode,
            "baseline_rmse": baseline_rmse,
            "baseline_r2": baseline_r2,
            "sce_rmse": best_sce_rmse,
            "sce_r2": summary.best_by_rmse.r2,
            "rmse_improvement_pct": rmse_improvement_pct,
            "r2_improvement_pp": float(summary.best_by_rmse.r2) - baseline_r2,
            "n_baseline_features": len(base_features),
            "n_sce_features": len(context_features),
            "model_preset": "search",
        },
        n_rows_loaded=n_rows_loaded,
        n_rows_after_filter=n_rows_after_filter,
        diagnostics={
            **_load_latest_diagnostics(config_name),
            "feature_dominance": feature_dominance,
        },
    )
    metadata["search"] = {
        "selection_protocol": {
            "candidate_eval_set": "validation",
            "winner_eval_set": "test",
            "val_fraction": search_val_fraction,
            "val_strategy": search_val_strategy,
        },
        "sampling_pct": sampling_pct_val,
        "min_configs": min_samples,
        "max_configs": max_samples,
        "model_presets": preset_names,
        "lm_threshold": lm_threshold,
        "lm_significant_base": sig_base,
        "lm_significant_context": sig_context,
        "best_by_rmse": summary.best_by_rmse.__dict__,
        "best_by_r2": summary.best_by_r2.__dict__,
    }
    with (output_dir / "data" / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    _record_failure_case(
        run_id=metadata["run_id"],
        config_name=config_name,
        model_type=model_type,
        rmse_improvement_pct=rmse_improvement_pct,
        runtime_seconds=runtime_seconds,
        baseline_runtime_seconds=baseline_runtime_seconds,
    )

    if run_report:
        from scripts.reporting import generate_search_reports

        generate_search_reports(results_df, output_dir)

    logger.info("Search results saved to: %s", output_dir)
    return output_dir


def run_experiment(
    config_name: str,
    sample_size: int | None = None,
    model_type_override: str | None = None,
    use_gpu_override: bool | None = None,
    context_variant_override: str | None = None,
    categorical_mode_override: str | None = None,
    cleanup: bool = False,
    run_grade: str = "exploratory",
    allow_dirty: bool = False,
) -> ExperimentResult:
    """
    Run SCE experiment on a single dataset.
    
    Key methodology:
    1. Split raw data into train/test first
    2. Fit SCE on the training partition only
    3. Transform the test partition with train-derived statistics only
    4. Train and evaluate the configured downstream model on the aligned baseline and enriched splits
    
    Args:
        config_name: Name of config file (without .toml)
        sample_size: Optional sample size for large datasets.
                     If None, uses smart defaults:
                     - UAE datasets: 100,000 samples
                     - Other datasets: no sampling
        
    Returns:
        ExperimentResult with metrics
    """
    _enforce_run_grade_policy(run_grade, allow_dirty)

    config_path = CONFIGS_DIR / f"{config_name}.toml"
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {config_name}")
    print(f"{'='*60}")
    logger.info(f"="*80)
    logger.info(f"STARTING EXPERIMENT: {config_name}")
    logger.info(f"="*80)

    start_time = time.time()
    
    # Load config and data
    config = load_config(config_path)
    logger.debug(f"Config loaded: {json.dumps(config, indent=2, default=str)}")
    context_variant = _resolve_context_variant(config, context_variant_override)
    categorical_mode = _resolve_categorical_mode(config, categorical_mode_override)
    context_label = get_context_variant_label(context_variant)
    model_type, model_params = _resolve_standard_model(
        config,
        model_type_override=model_type_override,
        use_gpu_override=use_gpu_override,
    )
    model_label = get_model_label(model_type)
    use_gpu = bool(model_params.get("use_gpu", False))

    _log_model_backend(model_type, use_gpu=use_gpu)
    
    df = load_dataset(config)
    n_rows_loaded = len(df)
    
    target_col = config["target"]["column"]
    logger.info(f"Target column: {target_col}")

    df = _filter_target_rows(df, target_col)
    resolved_sample_size = _resolve_sample_size(config_name, sample_size)
    df = _sample_dataset(df, resolved_sample_size)
    n_rows_after_filter = len(df)
    if resolved_sample_size and len(df) == resolved_sample_size:
        print(f"  Sampled to {resolved_sample_size:,} rows")
    
    print(f"  Dataset size: {len(df):,} rows")
    print(f"  Target: {target_col}")
    logger.info(f"Final dataset size: {len(df):,} rows")
    
    # Log target statistics (including all percentiles from config, with fallback)
    target_series = df[target_col]
    target_stats = {
        'count': len(target_series),
        'min': target_series.min(),
        'max': target_series.max(),
        'mean': target_series.mean(),
        'median': target_series.median(),
        'std': target_series.std(),
    }

    # Determine which percentiles to log: prefer config, fall back to defaults
    percentiles_from_config = None
    if isinstance(config, dict):
        percentiles_from_config = (
            config.get("evaluation", {}).get("percentiles")
            or config.get("metrics", {}).get("percentiles")
        )
    default_percentiles = [5, 10, 20, 25, 33, 66, 75, 80, 90, 95]
    percentiles = percentiles_from_config or default_percentiles

    for p in percentiles:
        try:
            p_float = float(p)
        except (TypeError, ValueError):
            continue
        # Expect p as 0-100; convert to 0-1 for quantile
        q = p_float / 100.0
        if not 0.0 <= q <= 1.0:
            continue
        # Use two-digit formatting (e.g., 5 -> q05) to match existing keys
        key = f"q{int(round(p_float)):02d}"
        target_stats[key] = target_series.quantile(q)
    logger.info(f"Target statistics: {json.dumps(target_stats, default=float)}")
    
    # ==== SPLIT FIRST, THEN FIT SCE ON TRAIN ONLY ====
    grouping_label = "manual grouping columns" if categorical_mode == "manual" else "auto-detected grouping columns"
    print(f"\n  Applying {context_label} enrichment ({grouping_label})...")
    logger.info("="*80)
    logger.info("APPLYING %s ENRICHMENT", context_label.upper())
    logger.info("="*80)

    split = _split_dataset(df, target_col, config=config)
    output_dir = _create_output_dir(config_name, "single")
    enrichment = _run_sce_enrichment(
        split.train_df,
        config,
        target_col,
        cleanup=cleanup,
        transform_df=split.test_df,
        context_variant=context_variant,
        categorical_mode=categorical_mode,
    )
    df_enriched = enrichment.df_enriched
    
    # Get SCE feature columns
    sce_feature_cols = enrichment.sce_feature_cols
    print(f"    Generated {len(sce_feature_cols)} SCE features")
    logger.info(f"Generated {len(sce_feature_cols)} SCE features")
    logger.debug(f"SCE feature columns: {sce_feature_cols[:10]}..." if len(sce_feature_cols) > 10 else f"SCE feature columns: {sce_feature_cols}")
    
    # ==== RATIO FEATURES DISABLED ====
    # Target-derived features are prohibited for evaluation.
    ratio_cols: list[str] = []
    logger.info("Ratio features disabled (target-derived features are not allowed)")

    all_sce_cols = enrichment.all_sce_cols
    logger.info(
        "Total enriched features: %s (SCE: %s, Ratios: %s)",
        len(all_sce_cols),
        len(sce_feature_cols),
        len(ratio_cols),
    )
    
    train_df = split.train_df
    test_df = split.test_df
    train_enriched = df_enriched.loc[split.train_idx]
    test_enriched = df_enriched.loc[split.test_idx]
    
    # ==== BASELINE MODEL (configured downstream model, no SCE features) ====
    print(f"\n  Training baseline {model_label} model...")
    logger.info("="*80)
    logger.info("BASELINE MODEL (NO SCE)")
    logger.info("="*80)
    
    train_base = prepare_features(train_df, config, target_col)
    test_base = prepare_features(
        test_df,
        config,
        target_col,
        encoder=train_base.encoder,
        droplist=train_base.droplist,
    )
    X_train_base, y_train = train_base.X, train_base.y
    X_test_base, y_test = test_base.X, test_base.y
    
    # Align columns
    X_train_base, X_test_base, common_cols = _align_common_columns(X_train_base, X_test_base)
    logger.debug(f"Common columns between train and test: {len(common_cols)}")
    X_train_base = X_train_base.fillna(0)
    X_test_base = X_test_base.fillna(0)
    
    logger.info(f"Baseline feature matrix: {X_train_base.shape[0]} x {X_train_base.shape[1]}")
    
    baseline_start = time.perf_counter()
    baseline_model = train_configured_model(X_train_base, y_train, model_type, model_params)
    baseline_preds = baseline_model.predict(X_test_base)
    baseline_runtime_seconds = time.perf_counter() - baseline_start
    
    # Log prediction statistics
    logger.debug(f"Baseline predictions: min={baseline_preds.min():.2f}, max={baseline_preds.max():.2f}, "
                f"mean={baseline_preds.mean():.2f}, std={baseline_preds.std():.2f}")
    
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
    baseline_r2 = r2_score(y_test, baseline_preds)
    
    print(f"    Baseline RMSE: {baseline_rmse:,.2f}")
    print(f"    Baseline R2:   {baseline_r2:.4f}")
    logger.info(f"Baseline RMSE: {baseline_rmse:,.2f}")
    logger.info(f"Baseline R2: {baseline_r2:.4f}")
    
    # Log per-sample errors for outlier detection
    baseline_errors = np.abs(y_test.values - baseline_preds)
    logger.debug(f"Baseline errors: min={baseline_errors.min():.2f}, max={baseline_errors.max():.2f}, "
                f"mean={baseline_errors.mean():.2f}, median={np.median(baseline_errors):.2f}")
    
    # Find worst predictions
    worst_indices = np.argsort(baseline_errors)[-5:]
    logger.debug("Top 5 worst baseline predictions:")
    for idx in worst_indices:
        actual = y_test.iloc[idx]
        pred = baseline_preds[idx]
        error = baseline_errors[idx]
        logger.debug(f"  Actual: {actual:.2f}, Predicted: {pred:.2f}, Error: {error:.2f}, "
                    f"Error %: {error/actual*100:.1f}%")
    
    # ==== SCE-ENRICHED MODEL (same downstream model with SCE features) ====
    print(f"\n  Training {context_label}-enriched {model_label} model...")
    logger.info("="*80)
    logger.info("%s-ENRICHED MODEL", context_label.upper())
    logger.info("="*80)
    
    train_sce = prepare_features(
        train_enriched,
        config,
        target_col,
        encoder=train_base.encoder,
        droplist=train_base.droplist,
    )
    test_sce = prepare_features(
        test_enriched,
        config,
        target_col,
        encoder=train_base.encoder,
        droplist=train_base.droplist,
    )
    X_train_sce, y_train_sce = train_sce.X, train_sce.y
    X_test_sce, y_test_sce = test_sce.X, test_sce.y
    
    # Add SCE and ratio features efficiently using concat (avoid DataFrame fragmentation)
    sce_train_cols = {col: train_enriched.loc[X_train_sce.index, col].values for col in all_sce_cols if col in train_enriched.columns}
    sce_test_cols = {col: test_enriched.loc[X_test_sce.index, col].values for col in all_sce_cols if col in test_enriched.columns}

    X_train_sce = pd.concat([X_train_sce, pd.DataFrame(sce_train_cols, index=X_train_sce.index)], axis=1)
    X_test_sce = pd.concat([X_test_sce, pd.DataFrame(sce_test_cols, index=X_test_sce.index)], axis=1)
    logger.info(f"Added {len(sce_train_cols)} SCE/ratio features to base features")
    
    # Drop rows with inf/NaN after enrichment (no imputation)
    X_train_sce = X_train_sce.replace([np.inf, -np.inf], np.nan)
    X_test_sce = X_test_sce.replace([np.inf, -np.inf], np.nan)

    train_before = len(X_train_sce)
    test_before = len(X_test_sce)
    run_cfg = config.get("run", {})
    min_non_missing_pct = run_cfg.get("min_non_missing_pct", 0.5)
    min_non_missing = max(1, int(min_non_missing_pct * X_train_sce.shape[1]))
    X_train_sce = X_train_sce.dropna(thresh=min_non_missing)
    X_test_sce = X_test_sce.dropna(thresh=min_non_missing)
    if len(X_train_sce) < train_before or len(X_test_sce) < test_before:
        logger.warning(
            "Dropped %s train and %s test rows after SCE enrichment due to missing values "
            "(min_non_missing=%s of %s features)",
            train_before - len(X_train_sce),
            test_before - len(X_test_sce),
            min_non_missing,
            X_train_sce.shape[1],
        )

    # Align baseline and SCE datasets to common indices for fair comparison
    common_train_idx = X_train_base.index.intersection(X_train_sce.index)
    common_test_idx = X_test_base.index.intersection(X_test_sce.index)
    X_train_base = X_train_base.loc[common_train_idx]
    y_train = y_train.loc[common_train_idx]
    X_test_base = X_test_base.loc[common_test_idx]
    y_test = y_test.loc[common_test_idx]
    X_train_sce = X_train_sce.loc[common_train_idx]
    y_train_sce = y_train_sce.loc[common_train_idx]
    X_test_sce = X_test_sce.loc[common_test_idx]
    y_test_sce = y_test_sce.loc[common_test_idx]
    
    # Align columns
    X_train_sce, X_test_sce, common_cols_sce = _align_common_columns(X_train_sce, X_test_sce)
    logger.debug(f"Common columns between train and test: {len(common_cols_sce)}")
    
    logger.info(f"SCE-enriched feature matrix: {X_train_sce.shape[0]} x {X_train_sce.shape[1]}")
    logger.info(f"Feature count increase: {X_train_sce.shape[1] - X_train_base.shape[1]} features "
               f"({(X_train_sce.shape[1] / X_train_base.shape[1] - 1) * 100:.1f}% increase)")
    
    sce_model = train_configured_model(X_train_sce, y_train_sce, model_type, model_params)
    sce_preds = sce_model.predict(X_test_sce)
    
    # Log prediction statistics
    logger.debug(f"SCE predictions: min={sce_preds.min():.2f}, max={sce_preds.max():.2f}, "
                f"mean={sce_preds.mean():.2f}, std={sce_preds.std():.2f}")
    
    sce_rmse = np.sqrt(mean_squared_error(y_test, sce_preds))
    sce_r2 = r2_score(y_test, sce_preds)
    
    print(f"    SCE RMSE: {sce_rmse:,.2f}")
    print(f"    SCE R2:   {sce_r2:.4f}")
    logger.info(f"SCE RMSE: {sce_rmse:,.2f}")
    logger.info(f"SCE R2: {sce_r2:.4f}")
    
    # Log per-sample errors for outlier detection
    sce_errors = np.abs(y_test.values - sce_preds)
    logger.debug(f"SCE errors: min={sce_errors.min():.2f}, max={sce_errors.max():.2f}, "
                f"mean={sce_errors.mean():.2f}, median={np.median(sce_errors):.2f}")
    
    # Find worst predictions
    worst_indices = np.argsort(sce_errors)[-5:]
    logger.debug("Top 5 worst SCE predictions:")
    for idx in worst_indices:
        actual = y_test.iloc[idx]
        pred = sce_preds[idx]
        error = sce_errors[idx]
        logger.debug(f"  Actual: {actual:.2f}, Predicted: {pred:.2f}, Error: {error:.2f}, "
                    f"Error %: {error/actual*100:.1f}%")
    
    # Compare prediction improvements
    error_reduction = baseline_errors - sce_errors
    improved_samples = (error_reduction > 0).sum()
    worsened_samples = (error_reduction < 0).sum()
    logger.info(f"Per-sample comparison: {improved_samples} improved, {worsened_samples} worsened, "
               f"{len(error_reduction) - improved_samples - worsened_samples} unchanged")
    
    # Compute improvements
    rmse_improvement = ((baseline_rmse - sce_rmse) / baseline_rmse) * 100
    r2_improvement = ((sce_r2 - baseline_r2) / max(abs(baseline_r2), 0.001)) * 100
    
    runtime = time.time() - start_time
    
    print("\n  Results:")
    print(f"    RMSE improvement: {rmse_improvement:+.2f}%")
    print(f"    R2 improvement:   {r2_improvement:+.2f}%")
    print(f"    Runtime: {runtime:.1f}s")
    
    logger.info("="*80)
    logger.info("FINAL RESULTS")
    logger.info("="*80)
    logger.info(f"RMSE improvement: {rmse_improvement:+.2f}%")
    logger.info(f"R2 improvement: {r2_improvement:+.2f}%")
    logger.info(f"Runtime: {runtime:.1f}s")
    logger.info(f"Experiment completed successfully for {config_name}")
    logger.info("="*80)

    sce_config = _build_sce_config(
        config,
        target_col,
        cleanup=cleanup,
        context_variant=context_variant,
        categorical_mode=categorical_mode,
    )

    # Diagnostics for report-grade promotion: file-based diagnostics are loaded
    # from results/diagnostics/<dataset>/, feature dominance is computed in-run
    # from the trained SCE model's importance.
    diagnostics = _load_latest_diagnostics(config_name)
    importance_df = extract_feature_importance(sce_model, X_train_sce.columns)
    importance_path = output_dir / "data" / "sce_feature_importance.csv"
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_path, index=False)
    diagnostics["feature_dominance"] = audit_feature_dominance(importance_path)

    metadata = _collect_run_metadata(
        config_name=config_name,
        config_path=config_path,
        config=config,
        run_grade=run_grade,
        split=split,
        sce_config=sce_config,
        model_type=model_type,
        model_params=model_params,
        runtime_seconds=runtime,
        metrics={
            "context_variant": context_variant,
            "categorical_mode": categorical_mode,
            "baseline_rmse": baseline_rmse,
            "baseline_r2": baseline_r2,
            "sce_rmse": sce_rmse,
            "sce_r2": sce_r2,
            "rmse_improvement_pct": rmse_improvement,
            "r2_improvement_pp": sce_r2 - baseline_r2,
            "n_baseline_features": len(X_train_base.columns),
            "n_sce_features": len(X_train_sce.columns),
            "model_preset": "default",
        },
        n_rows_loaded=n_rows_loaded,
        n_rows_after_filter=n_rows_after_filter,
        diagnostics=diagnostics,
    )
    with (output_dir / "data" / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    _record_failure_case(
        run_id=metadata["run_id"],
        config_name=config_name,
        model_type=model_type,
        rmse_improvement_pct=rmse_improvement,
        runtime_seconds=runtime,
        baseline_runtime_seconds=baseline_runtime_seconds,
    )
    
    return ExperimentResult(
        dataset=config_name,
        model_type=model_type,
        context_variant=context_variant,
        categorical_mode=categorical_mode,
        baseline_rmse=baseline_rmse,
        baseline_r2=baseline_r2,
        sce_rmse=sce_rmse,
        sce_r2=sce_r2,
        rmse_improvement_pct=rmse_improvement,
        r2_improvement_pct=r2_improvement,
        n_samples=len(df),
        n_baseline_features=len(X_train_base.columns),
        n_sce_features=len(X_train_sce.columns),
        runtime_seconds=runtime,
        metadata=metadata,
    )


def run_all(
    sample_size: int | None = None,
    use_search: bool = False,
    sampling_pct: float | None = None,
    p_threshold: float | None = None,
    run_report: bool = False,
    model_presets_override: list[str] | None = None,
    model_type_override: str | None = None,
    use_gpu_override: bool | None = None,
    context_variant_override: str | None = None,
    categorical_mode_override: str | None = None,
    cleanup: bool = False,
    run_grade: str = "exploratory",
    allow_dirty: bool = False,
) -> list[ExperimentResult]:
    """
    Run experiments on all configured datasets.
    
    Args:
        sample_size: Optional sample size for large datasets.
                     If None, uses smart defaults per dataset (100k for UAE).
    """
    results = []
    
    for config_path in sorted(CONFIGS_DIR.glob("*.toml")):
        config_name = config_path.stem
        try:
            if use_search:
                run_search_experiment(
                    config_name,
                    sample_size=sample_size,
                    sampling_pct=sampling_pct,
                    p_threshold=p_threshold,
                    run_report=True,
                    model_presets_override=model_presets_override,
                    model_type_override=model_type_override,
                    use_gpu_override=use_gpu_override,
                    context_variant_override=context_variant_override,
                    categorical_mode_override=categorical_mode_override,
                    cleanup=cleanup,
                    run_grade=run_grade,
                    allow_dirty=allow_dirty,
                )
                continue
            result = run_experiment(
                config_name,
                sample_size=sample_size,
                model_type_override=model_type_override,
                use_gpu_override=use_gpu_override,
                context_variant_override=context_variant_override,
                categorical_mode_override=categorical_mode_override,
                cleanup=cleanup,
                run_grade=run_grade,
                allow_dirty=allow_dirty,
            )
            results.append(result)
        except Exception as e:
            print(f"Error running {config_name}: {e}")
    
    return results


def save_results(results: list[ExperimentResult]) -> Path:
    """Save results to JSON and copy debug log to results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    results_path = RESULTS_DIR / "experiment_results.json"
    with results_path.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Copy debug log to results directory for GitHub Actions artifacts
    log_file = PROJECT_ROOT / "experiment_debug.log"
    if log_file.exists():
        import shutil
        dest_log = RESULTS_DIR / "experiment_debug.log"
        shutil.copy2(log_file, dest_log)
        print(f"Debug log saved to: {dest_log}")
        logger.info(f"Debug log copied to results directory: {dest_log}")
    
    return results_path


def run_categorical_mode_comparison(
    config_name: str,
    sample_size: int | None = None,
    model_type_override: str | None = None,
    use_gpu_override: bool | None = None,
    context_variant_override: str | None = None,
    cleanup: bool = False,
    run_grade: str = "exploratory",
    allow_dirty: bool = False,
) -> Path:
    """Run the same experiment twice and compare manual vs auto grouping modes."""
    manual_result = run_experiment(
        config_name,
        sample_size=sample_size,
        model_type_override=model_type_override,
        use_gpu_override=use_gpu_override,
        context_variant_override=context_variant_override,
        categorical_mode_override="manual",
        cleanup=cleanup,
        run_grade=run_grade,
        allow_dirty=allow_dirty,
    )
    auto_result = run_experiment(
        config_name,
        sample_size=sample_size,
        model_type_override=model_type_override,
        use_gpu_override=use_gpu_override,
        context_variant_override=context_variant_override,
        categorical_mode_override="auto",
        cleanup=cleanup,
        run_grade=run_grade,
        allow_dirty=allow_dirty,
    )

    output_dir = _create_output_dir(config_name, "categorical_compare")
    comparison_df = pd.DataFrame([asdict(manual_result), asdict(auto_result)])
    comparison_df.to_csv(output_dir / "data" / "categorical_mode_comparison.csv", index=False)

    metric_deltas = {
        "baseline_rmse_delta": auto_result.baseline_rmse - manual_result.baseline_rmse,
        "baseline_r2_delta": auto_result.baseline_r2 - manual_result.baseline_r2,
        "sce_rmse_delta": auto_result.sce_rmse - manual_result.sce_rmse,
        "sce_r2_delta": auto_result.sce_r2 - manual_result.sce_r2,
        "rmse_improvement_pct_delta": auto_result.rmse_improvement_pct - manual_result.rmse_improvement_pct,
        "r2_improvement_pct_delta": auto_result.r2_improvement_pct - manual_result.r2_improvement_pct,
        "sce_feature_count_delta": auto_result.n_sce_features - manual_result.n_sce_features,
    }
    metadata = {
        "dataset": config_name,
        "manual": asdict(manual_result),
        "auto": asdict(auto_result),
        "deltas_auto_minus_manual": metric_deltas,
    }
    with (output_dir / "data" / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    summary_lines = [
        f"# Categorical mode comparison: {config_name}",
        "",
        "| Mode | Baseline RMSE | SCE RMSE | RMSE Improvement % | Baseline R2 | SCE R2 | R2 Improvement % | Baseline Features | SCE Features | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| manual | {manual_result.baseline_rmse:,.2f} | {manual_result.sce_rmse:,.2f} | "
            f"{manual_result.rmse_improvement_pct:+.2f} | {manual_result.baseline_r2:.4f} | {manual_result.sce_r2:.4f} | "
            f"{manual_result.r2_improvement_pct:+.2f} | {manual_result.n_baseline_features} | {manual_result.n_sce_features} | "
            f"{manual_result.runtime_seconds:.1f} |"
        ),
        (
            f"| auto | {auto_result.baseline_rmse:,.2f} | {auto_result.sce_rmse:,.2f} | "
            f"{auto_result.rmse_improvement_pct:+.2f} | {auto_result.baseline_r2:.4f} | {auto_result.sce_r2:.4f} | "
            f"{auto_result.r2_improvement_pct:+.2f} | {auto_result.n_baseline_features} | {auto_result.n_sce_features} | "
            f"{auto_result.runtime_seconds:.1f} |"
        ),
        "",
        "## Auto minus manual",
        "",
        f"- SCE RMSE delta: {metric_deltas['sce_rmse_delta']:+,.2f}",
        f"- SCE R2 delta: {metric_deltas['sce_r2_delta']:+.4f}",
        f"- RMSE improvement delta: {metric_deltas['rmse_improvement_pct_delta']:+.2f} percentage points",
        f"- R2 improvement delta: {metric_deltas['r2_improvement_pct_delta']:+.2f} percentage points",
        f"- SCE feature count delta: {metric_deltas['sce_feature_count_delta']:+d}",
    ]
    (output_dir / "categorical_mode_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info("Categorical mode comparison saved to: %s", output_dir)
    return output_dir


def print_summary(results: list[ExperimentResult]) -> None:
    """Print summary table of results."""
    print("\n" + "="*140)
    print("EXPERIMENT SUMMARY")
    print("="*140)
    print(f"{'Dataset':<25} {'Model':<18} {'Context':<28} {'Mode':<8} {'Baseline RMSE':>15} {'SCE RMSE':>12} {'RMSE D%':>10} {'R2 D%':>10}")
    print("-"*140)
    
    for r in results:
        print(f"{r.dataset:<25} {get_model_label(r.model_type):<18} {get_context_variant_label(r.context_variant):<28} {r.categorical_mode:<8} {r.baseline_rmse:>15,.2f} {r.sce_rmse:>12,.2f} "
              f"{r.rmse_improvement_pct:>+10.2f} {r.r2_improvement_pct:>+10.2f}")
    
    print("-"*140)
    avg_rmse_imp = np.mean([r.rmse_improvement_pct for r in results])
    avg_r2_imp = np.mean([r.r2_improvement_pct for r in results])
    print(f"{'AVERAGE':<25} {'':<15} {'':<12} {avg_rmse_imp:>+10.2f} {avg_r2_imp:>+10.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SCE experiments")
    parser.add_argument("--dataset", "-d", help="Dataset config name to run")
    parser.add_argument("--all", "-a", action="store_true", help="Run all datasets")
    parser.add_argument("--sample-size", "-n", type=int, default=None,
                        help="Max sample size for large datasets (default: auto - 100k for UAE, full for others)")
    parser.add_argument("--search", action="store_true",
                        help="Run combinatorial model search")
    parser.add_argument("--sampling-pct", type=float, default=None,
                        help="Override sampling percentage for search (default: config)")
    parser.add_argument("--p-threshold", type=float, default=None,
                        help="Override LM p-value threshold for search (default: config)")
    parser.add_argument("--model-type", choices=SUPPORTED_MODEL_TYPES, default=None,
                        help="Override downstream model type")
    parser.add_argument("--context-variant", choices=SUPPORTED_CONTEXT_VARIANTS, default=None,
                        help="Override the context feature family: sce, target_mean, hierarchical_mean_count, or hierarchical_mean_std_count")
    parser.add_argument("--categorical-mode", choices=["manual", "auto"], default=None,
                        help="Override grouping-column selection for SCE: explicit config columns or auto-detection")
    parser.add_argument("--compare-categorical-modes", action="store_true",
                        help="Run the same standard experiment twice and compare manual vs auto grouping-column selection")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU acceleration for supported backends (xgboost, lightgbm, catboost)")
    parser.add_argument("--model-presets", nargs="*", default=None,
                        help="Override preset list for search runs")
    parser.add_argument("--xgb-presets", nargs="*", default=None,
                        help="Deprecated alias for --model-presets")
    parser.add_argument("--report", action="store_true",
                        help="Generate figures + tables for search results")
    parser.add_argument("--cleanup", action="store_true",
                        help="Enable feature correlation cleanup pipeline")
    parser.add_argument(
        "--run-grade",
        choices=["exploratory", "diagnostic", "report-grade"],
        default="exploratory",
        help="Run grade label used in metadata and quality gating",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow report-grade runs in dirty git state (warning only)",
    )
    parser.add_argument("--generate-figures", "-f", action="store_true",
                        help="Generate paper figures from results")
    
    args = parser.parse_args()
    model_presets_override = args.model_presets or args.xgb_presets

    if args.compare_categorical_modes and args.search:
        parser.error("--compare-categorical-modes is supported only for standard experiments, not --search runs")
    if args.compare_categorical_modes and not args.dataset:
        parser.error("--compare-categorical-modes requires --dataset")
    if args.compare_categorical_modes and args.categorical_mode is not None:
        parser.error("--compare-categorical-modes already runs both manual and auto, so do not combine it with --categorical-mode")
    
    if args.dataset:
        if args.compare_categorical_modes:
            output_dir = run_categorical_mode_comparison(
                args.dataset,
                sample_size=args.sample_size,
                model_type_override=args.model_type,
                use_gpu_override=True if args.use_gpu else None,
                context_variant_override=args.context_variant,
                cleanup=args.cleanup,
                run_grade=args.run_grade,
                allow_dirty=args.allow_dirty,
            )
            print(f"\nCategorical mode comparison saved to: {output_dir}")
            return 0

        if args.search:
            run_search_experiment(
                args.dataset,
                sample_size=args.sample_size,
                sampling_pct=args.sampling_pct,
                p_threshold=args.p_threshold,
                run_report=True,
                model_presets_override=model_presets_override,
                model_type_override=args.model_type,
                use_gpu_override=True if args.use_gpu else None,
                context_variant_override=args.context_variant,
                categorical_mode_override=args.categorical_mode,
                cleanup=args.cleanup,
                run_grade=args.run_grade,
                allow_dirty=args.allow_dirty,
            )
            return 0

        result = run_experiment(
            args.dataset,
            sample_size=args.sample_size,
            model_type_override=args.model_type,
            use_gpu_override=True if args.use_gpu else None,
            context_variant_override=args.context_variant,
            categorical_mode_override=args.categorical_mode,
            cleanup=args.cleanup,
            run_grade=args.run_grade,
            allow_dirty=args.allow_dirty,
        )
        save_results([result])
        return 0
    
    if args.all:
        results = run_all(
            sample_size=args.sample_size,
            use_search=args.search,
            sampling_pct=args.sampling_pct,
            p_threshold=args.p_threshold,
            run_report=args.report,
            model_presets_override=model_presets_override,
            model_type_override=args.model_type,
            use_gpu_override=True if args.use_gpu else None,
            context_variant_override=args.context_variant,
            categorical_mode_override=args.categorical_mode,
            cleanup=args.cleanup,
            run_grade=args.run_grade,
            allow_dirty=args.allow_dirty,
        )
        if results:
            save_results(results)
            print_summary(results)
        return 0
    
    if args.generate_figures:
        print("Figure generation not yet implemented")
        return 1
    
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
