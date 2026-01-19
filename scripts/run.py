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
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
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
    resolve_xgboost_presets,
    aggregate_importance,
    run_iterative_pruning,
)
from sce.config import AggregationMethod

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


def _log_xgboost_version() -> None:
    try:
        import xgboost as xgb
        logger.info("XGBoost version: %s", getattr(xgb, "__version__", "unknown"))
    except Exception as exc:
        logger.warning("XGBoost not available: %s", exc)


def _log_sce_equations() -> None:
    logger.info("SCE Eq(1): phi_k(x_t) = S_k({y_s : s in N_k(t)})")
    logger.info("SCE Eq(2): Phi(x_t) = [phi^(1)(x_t), ..., phi^(K)(x_t)]")
    logger.info("SCE Eq(3): r_k,z = (y_t - mu_k) / (sigma_k + eps), r_k,ratio = y_t / (median_k + eps)")
    logger.info("SCE Eq(4): phi_cf^(k)(x_t) = S_k({y_s : s in N_k(t) \ I_m})")


@dataclass
class ExperimentResult:
    """Container for experiment metrics."""
    dataset: str
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


def load_config(config_path: Path) -> dict[str, Any]:
    """Load TOML configuration file."""
    with config_path.open("rb") as f:
        return tomllib.load(f)


def load_dataset(config: dict[str, Any]) -> pd.DataFrame:
    """Load dataset based on config. Downloads from Hugging Face if needed."""
    data_path = PROJECT_ROOT / config["dataset"]["path"]
    logger.info(f"Loading dataset from: {data_path}")

    # Check if dataset needs to be downloaded
    if not data_path.exists():
        source = config["dataset"].get("source", "local")
        if source == "remote":
            logger.info("Dataset not found locally, downloading from Hugging Face...")
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
    target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix and target vector."""
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

    removed_cols: list[tuple[str, str, float]] = []
    for col in feature_cols:
        series = working_df[col]
        missing_rate = float(series.isna().mean())
        if missing_rate > missing_threshold:
            removed_cols.append((col, "missing_rate", missing_rate))
            continue
        if drop_zero_variance:
            nunique = series.nunique(dropna=True)
            if nunique <= 1:
                removed_cols.append((col, "zero_variance", float(nunique)))

    if removed_cols:
        removed_set = {col for col, _, _ in removed_cols}
        numeric_cols = [c for c in numeric_cols if c not in removed_set]
        categorical_cols = [c for c in categorical_cols if c not in removed_set]
        feature_cols = numeric_cols + categorical_cols
        working_df = working_df[feature_cols + [target_col]].copy()

        logger.warning(
            "Pruned %s feature columns (missing_threshold=%.2f, drop_zero_variance=%s)",
            len(removed_cols),
            missing_threshold,
            drop_zero_variance,
        )
        for col, reason, value in removed_cols:
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

    # Encode categoricals
    for col in categorical_cols:
        if X[col].dtype == "object":
            original_unique = X[col].nunique()
            X[col] = pd.Categorical(X[col]).codes
            logger.debug(f"Encoded categorical '{col}': {original_unique} unique values")

    y = working_df[target_col].copy()
    logger.info(f"Target '{target_col}': min={y.min():.2f}, max={y.max():.2f}, "
                f"mean={y.mean():.2f}, median={y.median():.2f}, std={y.std():.2f}")
    
    return X, y


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


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, config: dict[str, Any]):
    """Train XGBoost model with config parameters."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        model_cfg = config.get("model", {})
        return GradientBoostingRegressor(
            n_estimators=model_cfg.get("n_estimators", 100),
            max_depth=model_cfg.get("max_depth", 6),
            learning_rate=model_cfg.get("learning_rate", 0.1),
            random_state=42
        ).fit(X_train, y_train)
    
    model_cfg = config.get("model", {})
    model = XGBRegressor(
        n_estimators=model_cfg.get("n_estimators", 100),
        max_depth=model_cfg.get("max_depth", 6),
        learning_rate=model_cfg.get("learning_rate", 0.1),
        random_state=42,
        verbosity=0
    )
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
    xgb_presets_override: list[str] | None = None,
    cleanup: bool = False,
) -> Path:
    """Run combinatorial search + reporting pipeline."""
    config_path = CONFIGS_DIR / f"{config_name}.toml"
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        raise FileNotFoundError(f"Config not found: {config_path}")

    logger.info("=" * 80)
    logger.info("STARTING SEARCH EXPERIMENT: %s", config_name)
    logger.info("=" * 80)

    _log_xgboost_version()

    config = load_config(config_path)
    run_cfg = config.get("run", {})
    search_cfg = run_cfg.get("search", {})
    lm_cfg = run_cfg.get("lm_selection", {})

    df = load_dataset(config)
    target_col = config["target"]["column"]

    df = df.dropna(subset=[target_col])
    df = df[df[target_col] > 0]

    # Determine sample size with smart defaults (align with run_experiment)
    if sample_size is None:
        if "uae" in config_name.lower() or "dubai" in config_name.lower():
            sample_size = 50000

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info("Sampled dataset to %s rows", sample_size)

    output_dir = _create_output_dir(config_name, "search")

    # SCE enrichment (leakage-safe, cross-fitted)
    _log_sce_equations()
    sce_cfg = config.get("sce", {})
    agg_names = sce_cfg.get("aggregations", ["mean", "std", "median", "count"])
    aggregations = [AggregationMethod(name) for name in agg_names]
    manual_categoricals = config.get("features", {}).get("categorical", [])

    use_cross_fitting = sce_cfg.get("use_cross_fitting", True)
    include_fold_variance = sce_cfg.get("include_fold_variance", True)
    fold_variance_features = sce_cfg.get("fold_variance_features", ["std", "lower", "upper"])
    add_backoff_depth = sce_cfg.get("add_backoff_depth", False)
    include_interactions = sce_cfg.get("include_interactions", True)
    max_interaction_depth = sce_cfg.get("max_interaction_depth", 2)
    cleanup_config = CleanupConfig() if cleanup else None
    logger.info(
        "SCE config: cross_fitting=%s, folds=%s, fold_variance=%s, variance_features=%s, backoff_depth=%s",
        use_cross_fitting,
        sce_cfg.get("n_folds", 5),
        include_fold_variance,
        fold_variance_features,
        add_backoff_depth,
    )
    sce_config = ContextConfig(
        target_col=target_col,
        categorical_cols=manual_categoricals,
        min_categorical_columns=sce_cfg.get("min_categorical_columns", 1),
        aggregations=aggregations,
        min_group_size=sce_cfg.get("min_group_size", 3),
        use_cross_fitting=use_cross_fitting,
        n_folds=sce_cfg.get("n_folds", 5),
        include_interactions=include_interactions,
        max_interaction_depth=max_interaction_depth,
        max_cardinality=sce_cfg.get("max_cardinality", 100),
        include_fold_variance=include_fold_variance,
        fold_variance_features=fold_variance_features,
        add_backoff_depth=add_backoff_depth,
        cleanup_config=cleanup_config,
    )

    engine = StatisticalContextEngine(sce_config)
    df_enriched = engine.fit_transform(df)

    if getattr(engine, "_cleanup_report", None):
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

    backoff_cols = [c for c in df_enriched.columns if c.endswith("_backoff_depth")]
    if backoff_cols:
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

    sce_feature_cols = [c for c in df_enriched.columns if c not in df.columns]

    # Train/test split
    train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]
    train_enriched = df_enriched.loc[train_idx]
    test_enriched = df_enriched.loc[test_idx]

    X_train_base, y_train = prepare_features(train_df, config, target_col)
    X_test_base, y_test = prepare_features(test_df, config, target_col)

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
    if xgb_presets_override:
        run_cfg = dict(run_cfg)
        run_cfg["xgboost_configs"] = xgb_presets_override
    preset_names, preset_params = resolve_xgboost_presets(run_cfg, config.get("model", {}))

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

    searcher = FeatureCombinationSearch(
        base_features=base_features,
        context_features=context_features,
        sampling_pct=sampling_pct_val,
        min_samples=min_samples,
        max_samples=max_samples,
        model_configs=preset_names,
        model_params=preset_params,
        run_ablation=run_ablation,
        run_significance_selection=run_significance,
        p_threshold=lm_threshold,
    )

    summary = searcher.search(X_train_all, y_train, X_test_all, y_test)

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
            "features": "|".join(r.features),
        }
        for r in summary.all_results
    ])
    results_df.to_csv(output_dir / "data" / "model_comparison.csv", index=False)

    # Importance aggregation
    agg_importance = aggregate_importance(summary.all_results)
    if not agg_importance.empty:
        agg_importance.to_csv(output_dir / "data" / "aggregated_feature_importance.csv", index=False)

    # Pruning steps (default preset)
    pruning_results, removed_df = run_iterative_pruning(
        X_train_all,
        y_train,
        X_test_all,
        y_test,
        features=base_features + context_features,
        model_config_name="default",
        model_params=preset_params,
    )
    pd.DataFrame([r.__dict__ for r in pruning_results]).to_csv(
        output_dir / "data" / "xgb_pruning_trace.csv",
        index=False,
    )
    removed_df.to_csv(output_dir / "data" / "xgb_pruning_removed_features.csv", index=False)

    # Metadata summary
    metadata = {
        "dataset": config_name,
        "n_samples": len(df),
        "n_base_features": len(base_features),
        "n_context_features": len(context_features),
        "sampling_pct": sampling_pct_val,
        "min_configs": min_samples,
        "max_configs": max_samples,
        "xgboost_presets": preset_names,
        "lm_threshold": lm_threshold,
        "lm_significant_base": sig_base,
        "lm_significant_context": sig_context,
        "best_by_rmse": summary.best_by_rmse.__dict__,
        "best_by_r2": summary.best_by_r2.__dict__,
    }
    with (output_dir / "data" / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    if run_report:
        from scripts.reporting import generate_search_reports

        generate_search_reports(results_df, output_dir)

    logger.info("Search results saved to: %s", output_dir)
    return output_dir


def run_experiment(
    config_name: str,
    sample_size: int | None = None,
    cleanup: bool = False,
) -> ExperimentResult:
    """
    Run SCE experiment on a single dataset.
    
    Key methodology (matching paper):
    1. Apply SCE to FULL dataset first (maximizes context information)
    2. Create ratio features (Eq. 3) for relative comparisons
    3. Then split into train/test
    4. Use XGBoost (handles non-linear feature interactions)
    
    Args:
        config_name: Name of config file (without .toml)
        sample_size: Optional sample size for large datasets.
                     If None, uses smart defaults:
                     - UAE datasets: 100,000 samples
                     - Other datasets: no sampling
        
    Returns:
        ExperimentResult with metrics
    """
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

    _log_xgboost_version()
    
    start_time = time.time()
    
    # Load config and data
    config = load_config(config_path)
    logger.debug(f"Config loaded: {json.dumps(config, indent=2, default=str)}")
    
    df = load_dataset(config)
    
    target_col = config["target"]["column"]
    logger.info(f"Target column: {target_col}")
    
    # Drop rows with missing target
    initial_rows = len(df)
    missing_target = df[target_col].isna().sum()
    logger.debug(f"Missing target values: {missing_target}")
    df = df.dropna(subset=[target_col])
    
    zero_or_neg = (df[target_col] <= 0).sum()
    logger.debug(f"Zero or negative target values: {zero_or_neg}")
    df = df[df[target_col] > 0]
    
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        logger.warning(
            f"Dropped {rows_dropped} rows ({rows_dropped/initial_rows*100:.1f}%) "
            f"due to missing or invalid target values"
        )

    # Determine sample size with smart defaults
    if sample_size is None:
        # Use 50k for UAE datasets as they're very large
        if "uae" in config_name.lower() or "dubai" in config_name.lower():
            sample_size = 50000

    # Sample if needed - do this FIRST
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"  Sampled to {sample_size:,} rows")
        logger.info(f"Sampled dataset to {sample_size:,} rows")
    
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
    
    # ==== APPLY SCE TO FULL DATASET FIRST ====
    # This is crucial - SCE context is computed on ALL data before splitting
    print("\n  Applying SCE enrichment (auto-detection)...")
    logger.info("="*80)
    logger.info("APPLYING SCE ENRICHMENT")
    logger.info("="*80)
    
    _log_sce_equations()
    sce_cfg = config.get("sce", {})
    agg_names = sce_cfg.get("aggregations", ["mean", "std", "median", "count"])
    aggregations = [AggregationMethod(name) for name in agg_names]
    logger.info(f"Aggregation methods: {agg_names}")
    
    # Use categorical columns from features config (not sce section)
    # BUGFIX: Previously looked in config['sce']['categorical_cols'] which was None,
    # triggering auto-detection that misclassified low-cardinality numeric features
    # (e.g., capacity 1-16) as categorical. Now uses same list as baseline model.
    manual_categoricals = config.get("features", {}).get("categorical", [])
    logger.info(f"Manual categorical columns: {manual_categoricals}")
    
    use_cross_fitting = sce_cfg.get("use_cross_fitting", True)
    include_ratio_features = False
    include_fold_variance = sce_cfg.get("include_fold_variance", True)
    fold_variance_features = sce_cfg.get("fold_variance_features", ["std", "lower", "upper"])
    add_backoff_depth = sce_cfg.get("add_backoff_depth", False)
    include_interactions = sce_cfg.get("include_interactions", True)
    max_interaction_depth = sce_cfg.get("max_interaction_depth", 2)
    cleanup_config = CleanupConfig() if cleanup else None

    sce_config = ContextConfig(
        target_col=target_col,
        categorical_cols=manual_categoricals,  # Use features from config
        min_categorical_columns=sce_cfg.get("min_categorical_columns", 1),
        aggregations=aggregations,
        min_group_size=sce_cfg.get("min_group_size", 3),
        use_cross_fitting=use_cross_fitting,
        n_folds=sce_cfg.get("n_folds", 5),
        include_interactions=include_interactions,
        max_interaction_depth=max_interaction_depth,
        max_cardinality=sce_cfg.get("max_cardinality", 100),
        include_fold_variance=include_fold_variance,
        fold_variance_features=fold_variance_features,
        add_backoff_depth=add_backoff_depth,
        cleanup_config=cleanup_config,
    )
    logger.debug(
        "SCE Config: min_group_size=%s, use_cross_fitting=%s, include_interactions=%s, "
        "max_cardinality=%s, include_ratio_features=%s",
        sce_config.min_group_size,
        sce_config.use_cross_fitting,
        sce_config.include_interactions,
        sce_config.max_cardinality,
        include_ratio_features,
    )
    logger.info(
        "SCE config: cross_fitting=%s, folds=%s, fold_variance=%s, variance_features=%s, backoff_depth=%s",
        use_cross_fitting,
        sce_config.n_folds,
        include_fold_variance,
        fold_variance_features,
        add_backoff_depth,
    )
    
    engine = StatisticalContextEngine(sce_config)
    df_enriched = engine.fit_transform(df)

    if getattr(engine, "_cleanup_report", None):
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

    backoff_cols = [c for c in df_enriched.columns if c.endswith("_backoff_depth")]
    if backoff_cols:
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
    
    # Get SCE feature columns
    sce_feature_cols = [c for c in df_enriched.columns if c not in df.columns]
    print(f"    Generated {len(sce_feature_cols)} SCE features")
    logger.info(f"Generated {len(sce_feature_cols)} SCE features")
    logger.debug(f"SCE feature columns: {sce_feature_cols[:10]}..." if len(sce_feature_cols) > 10 else f"SCE feature columns: {sce_feature_cols}")
    
    # ==== RATIO FEATURES DISABLED ====
    # Target-derived features are prohibited for evaluation.
    ratio_cols: list[str] = []
    logger.info("Ratio features disabled (target-derived features are not allowed)")

    all_sce_cols = sce_feature_cols
    logger.info(
        "Total enriched features: %s (SCE: %s, Ratios: %s)",
        len(all_sce_cols),
        len(sce_feature_cols),
        len(ratio_cols),
    )
    
    # ==== NOW SPLIT (after SCE enrichment) ====
    # Use same indices for both baseline and enriched
    logger.info("="*80)
    logger.info("TRAIN/TEST SPLIT")
    logger.info("="*80)
    train_idx, test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42
    )
    logger.info(f"Train size: {len(train_idx)} ({len(train_idx)/len(df)*100:.1f}%)")
    logger.info(f"Test size: {len(test_idx)} ({len(test_idx)/len(df)*100:.1f}%)")
    
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]
    train_enriched = df_enriched.loc[train_idx]
    test_enriched = df_enriched.loc[test_idx]
    
    # Log train/test target distributions and basic similarity metrics
    train_target = train_df[target_col]
    test_target = test_df[target_col]

    train_stats = {
        "min": float(train_target.min()),
        "max": float(train_target.max()),
        "mean": float(train_target.mean()),
        "std": float(train_target.std()),
    }
    test_stats = {
        "min": float(test_target.min()),
        "max": float(test_target.max()),
        "mean": float(test_target.mean()),
        "std": float(test_target.std()),
    }

    similarity = {
        "mean_abs_diff": abs(train_stats["mean"] - test_stats["mean"]),
        "std_abs_diff": abs(train_stats["std"] - test_stats["std"]),
    }

    logger.info(
        "Target distribution train/test summary: %s",
        json.dumps(
            {
                "train": train_stats,
                "test": test_stats,
                "similarity": similarity,
            }
        ),
    )
    
    # ==== BASELINE MODEL (XGBoost, no SCE features) ====
    print("\n  Training baseline XGBoost model...")
    logger.info("="*80)
    logger.info("BASELINE MODEL (NO SCE)")
    logger.info("="*80)
    
    X_train_base, y_train = prepare_features(train_df, config, target_col)
    X_test_base, y_test = prepare_features(test_df, config, target_col)
    
    # Align columns
    common_cols = list(set(X_train_base.columns) & set(X_test_base.columns))
    logger.debug(f"Common columns between train and test: {len(common_cols)}")
    X_train_base = X_train_base[common_cols].fillna(0)
    X_test_base = X_test_base[common_cols].fillna(0)
    
    logger.info(f"Baseline feature matrix: {X_train_base.shape[0]} x {X_train_base.shape[1]}")
    
    baseline_model = train_xgboost(X_train_base, y_train, config)
    baseline_preds = baseline_model.predict(X_test_base)
    
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
    
    # ==== SCE-ENRICHED MODEL (XGBoost with SCE + ratio features) ====
    print("\n  Training SCE-enriched XGBoost model...")
    logger.info("="*80)
    logger.info("SCE-ENRICHED MODEL")
    logger.info("="*80)
    
    X_train_sce, y_train_sce = prepare_features(train_enriched, config, target_col)
    X_test_sce, y_test_sce = prepare_features(test_enriched, config, target_col)
    
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
    common_cols_sce = list(set(X_train_sce.columns) & set(X_test_sce.columns))
    logger.debug(f"Common columns between train and test: {len(common_cols_sce)}")
    X_train_sce = X_train_sce[common_cols_sce]
    X_test_sce = X_test_sce[common_cols_sce]
    
    logger.info(f"SCE-enriched feature matrix: {X_train_sce.shape[0]} x {X_train_sce.shape[1]}")
    logger.info(f"Feature count increase: {X_train_sce.shape[1] - X_train_base.shape[1]} features "
               f"({(X_train_sce.shape[1] / X_train_base.shape[1] - 1) * 100:.1f}% increase)")
    
    sce_model = train_xgboost(X_train_sce, y_train_sce, config)
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
    
    return ExperimentResult(
        dataset=config_name,
        baseline_rmse=baseline_rmse,
        baseline_r2=baseline_r2,
        sce_rmse=sce_rmse,
        sce_r2=sce_r2,
        rmse_improvement_pct=rmse_improvement,
        r2_improvement_pct=r2_improvement,
        n_samples=len(df),
        n_baseline_features=len(X_train_base.columns),
        n_sce_features=len(X_train_sce.columns),
        runtime_seconds=runtime
    )


def run_all(
    sample_size: int | None = None,
    use_search: bool = False,
    sampling_pct: float | None = None,
    p_threshold: float | None = None,
    run_report: bool = False,
    xgb_presets_override: list[str] | None = None,
    cleanup: bool = False,
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
                    xgb_presets_override=xgb_presets_override,
                    cleanup=cleanup,
                )
                continue
            result = run_experiment(config_name, sample_size=sample_size, cleanup=cleanup)
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


def print_summary(results: list[ExperimentResult]) -> None:
    """Print summary table of results."""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Dataset':<25} {'Baseline RMSE':>15} {'SCE RMSE':>12} {'RMSE D%':>10} {'R2 D%':>10}")
    print("-"*80)
    
    for r in results:
        print(f"{r.dataset:<25} {r.baseline_rmse:>15,.2f} {r.sce_rmse:>12,.2f} "
              f"{r.rmse_improvement_pct:>+10.2f} {r.r2_improvement_pct:>+10.2f}")
    
    print("-"*80)
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
    parser.add_argument("--xgb-presets", nargs="*", default=None,
                        help="Override XGBoost preset list (e.g., shallow boosted default)")
    parser.add_argument("--report", action="store_true",
                        help="Generate figures + tables for search results")
    parser.add_argument("--cleanup", action="store_true",
                        help="Enable feature correlation cleanup pipeline")
    parser.add_argument("--generate-figures", "-f", action="store_true",
                        help="Generate paper figures from results")
    
    args = parser.parse_args()
    
    if args.dataset:
        if args.search:
            run_search_experiment(
                args.dataset,
                sample_size=args.sample_size,
                sampling_pct=args.sampling_pct,
                p_threshold=args.p_threshold,
                run_report=True,
                xgb_presets_override=args.xgb_presets,
                cleanup=args.cleanup,
            )
            return 0

        result = run_experiment(args.dataset, sample_size=args.sample_size, cleanup=args.cleanup)
        save_results([result])
        return 0
    
    if args.all:
        results = run_all(
            sample_size=args.sample_size,
            use_search=args.search,
            sampling_pct=args.sampling_pct,
            p_threshold=args.p_threshold,
            run_report=args.report,
            xgb_presets_override=args.xgb_presets,
            cleanup=args.cleanup,
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
