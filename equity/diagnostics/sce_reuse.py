"""
@module: equity.diagnostics.sce_reuse
@depends: numpy, pandas, sklearn, equity.sce, sce
@exports: evaluate_equity_sce, run_permuted_target_equity,
          run_shuffled_groups_equity, run_crossfit_ab_equity,
          audit_feature_dominance_equity
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §8.1 (S4.6 reuse)
@data_flow: equity features frame -> EquityContextEnricher (+ optional CF toggle)
            -> baseline vs SCE Ridge model comparison (RMSE/R2)
            -> thin wrappers (permuted_target / shuffled_groups / crossfit_ab /
               feature_dominance) mirroring scripts/diagnostics/*

S4.6 equity-local SCE reuse runner. This is the equity adapter that does what
``scripts.diagnostics._common.evaluate_config_dataframe`` does for TOML-config
datasets, but against an equity features frame produced by
:func:`equity.features.build_features` + :class:`~equity.sce.EquityContextEnricher`.

Baseline-vs-SCE comparison semantics (mirrored from
``scripts/diagnostics/_common.py:evaluate_config_dataframe``):

1. Time-ordered train/test split of the (prepared) features frame.
2. Run SCE enrichment on the **train** split (``fit_transform``); ``transform``
   the **test** split with the fitted engine (no test leakage into stats).
3. **Baseline model**: fit on numeric non-SCE feature columns of train;
   predict test → ``baseline_rmse`` / ``baseline_r2``.
4. **SCE model**: fit on the same baseline columns PLUS SCE context columns;
   predict test → ``sce_rmse`` / ``sce_r2``.
5. Both legs use the **same model class** so the delta isolates the SCE context
   contribution, not a model-class confounds.

Model choice (documented deviation from scripts/run xgboost defaults)
---------------------------------------------------------------------
This equity runner uses ``sklearn.linear_model.Ridge(alpha=1.0)`` for BOTH
baseline and SCE legs. Rationale: (a) the S4.6 goal is the adapter surface,
not reproducing xgb training exactly; (b) Ridge keeps the smoke path
dependency-light and deterministic. Fairness is preserved because both legs
share the same estimator. Swap later if a heavier model is needed for a
report-grade A/B.

The four diagnostics thin wrappers mirror the scripts counterparts:

* :func:`run_permuted_target_equity` — permute target, average SCE advantage;
  ``pass = permuted_mean_advantage < 1.0``.
* :func:`run_shuffled_groups_equity` — shuffle categorical labels;
  ``pass = real_advantage >= 50% of (real - shuffled)`` (same threshold math
  as ``scripts/diagnostics/shuffled_groups.py``).
* :func:`run_crossfit_ab_equity` — CF on vs off;
  ``leakage_signal_pp = (rmse_no_cf - rmse_cf) / rmse_no_cf * 100``.
* :func:`audit_feature_dominance_equity` — re-exports
  :func:`scripts.run.audit_feature_dominance` (no duplicate).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

from equity.sce import EquityContextEnricher, EquityHierarchyConfig
from equity.sce.enrich import _level_from_context_column
from sce import StatisticalContextEngine

# Non-feature / identity columns that never enter the Ridge design matrix.
_IDENTITY_COLS = frozenset(
    {
        "ticker",
        "sector",
        "industry",
        "mktcap_bucket",
        "time_bucket",
    }
)


def _numeric_feature_cols(
    df: pd.DataFrame,
    *,
    target_col: str,
    time_col: str,
    extra_exclude: Iterable[str] = (),
) -> list[str]:
    """Select numeric columns suitable as model features (exclude target/time/id)."""
    exclude = set(_IDENTITY_COLS) | {target_col, time_col} | set(extra_exclude)
    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def _sce_context_cols(df: pd.DataFrame, target_col: str) -> list[str]:
    """Columns that parse as SCE context features under the equity naming contract."""
    return [c for c in df.columns if _level_from_context_column(c, target_col) is not None]


def _fit_predict_rmse_r2(
    model: Ridge,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[float, float]:
    """Fit Ridge, predict test, return (rmse, r2). Constant-y → r2=0.0."""
    if X_train.empty or X_test.empty or len(y_train) == 0 or len(y_test) == 0:
        return float("nan"), float("nan")
    model.fit(X_train.fillna(0.0).to_numpy(dtype=float), y_train.to_numpy(dtype=float))
    preds = model.predict(X_test.fillna(0.0).to_numpy(dtype=float))
    y_true = y_test.to_numpy(dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
    # r2_score raises on constant y; treat that as a zero-skill baseline.
    if np.unique(y_true).size < 2:
        r2 = 0.0
    else:
        r2 = float(r2_score(y_true, preds))
    return rmse, r2


def evaluate_equity_sce(
    features: pd.DataFrame,
    target_col: str = "ret_1d",
    *,
    hierarchy: EquityHierarchyConfig | None = None,
    sectors: pd.DataFrame | Path | str | None = None,
    use_cross_fitting: bool | None = None,
    test_frac: float = 0.3,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run baseline + SCE on an equity features frame; return RMSE/R2 metrics.

    Mirrors ``scripts.diagnostics._common.evaluate_config_dataframe`` semantics
    with a lighter Ridge estimator (see module docstring).

    Parameters
    ----------
    features:
        Output of :func:`equity.features.build_features` (or a synthetic panel
        with the same schema). Must carry ``ticker``, the hierarchy ``time_col``,
        and either ``target_col`` or ``ret_1d_log`` (aliased by the enricher).
    target_col:
        Target column name (default ``"ret_1d"``).
    hierarchy:
        Frozen equity hierarchy knobs. Defaults to :class:`EquityHierarchyConfig`.
    sectors:
        Static hierarchy map (DataFrame / path / ``None`` = default CSV).
    use_cross_fitting:
        Override SCE cross-fitting. ``None`` keeps the hierarchy default
        (rolling CF on). ``False`` forces non-CF fit for the A/B diagnostic.
    test_frac:
        Fraction of rows (time-ordered) held out as the test split.
    random_state:
        Seed for the Ridge estimator (determinism only; split is time-ordered).

    Returns
    -------
    dict
        ``{baseline_rmse, baseline_r2, sce_rmse, sce_r2, n_features_sce,
        n_features_baseline, n_train, n_test, use_cross_fitting}``.
    """
    if hierarchy is None:
        hierarchy = EquityHierarchyConfig(target_col=target_col)
    elif hierarchy.target_col != target_col:
        # Honour the explicit target_col arg; rebuild hierarchy with override.
        hierarchy = replace(hierarchy, target_col=target_col)

    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1); got {test_frac}")

    enricher = EquityContextEnricher(hierarchy=hierarchy, sectors=sectors)
    prepared = enricher._prepare(features)
    prepared = prepared.loc[prepared[target_col].notna()].copy()
    if prepared.empty:
        raise ValueError("evaluate_equity_sce: no rows with non-null target after prepare.")

    time_col = hierarchy.time_col
    prepared = prepared.sort_values(time_col).reset_index(drop=True)
    n = len(prepared)
    split_idx = max(1, min(n - 1, int(round(n * (1.0 - test_frac)))))
    train_df = prepared.iloc[:split_idx].copy()
    test_df = prepared.iloc[split_idx:].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"evaluate_equity_sce: empty train/test after split "
            f"(n={n}, split_idx={split_idx}, test_frac={test_frac})."
        )

    cfg = enricher.build_context_config()
    if use_cross_fitting is not None:
        cfg = replace(
            cfg,
            use_cross_fitting=bool(use_cross_fitting),
            cross_fit_strategy="rolling" if use_cross_fitting else "off",
        )
    effective_cf = bool(cfg.use_cross_fitting)

    engine = StatisticalContextEngine(cfg)
    # CF path: fit_transform trains OOF context on train; fit() side-effect still
    # builds an in-sample _stats_dict used by subsequent transform() on test
    # (see equity.sce.enrich / sce.engine citations in transform_partial).
    train_enr = engine.fit_transform(train_df)
    test_enr = engine.transform(test_df)
    train_enr = enricher._post_filter_interactions(train_enr)
    test_enr = enricher._post_filter_interactions(test_enr)

    sce_cols = _sce_context_cols(train_enr, target_col)
    # Baseline features = numeric columns present on the *prepared* (pre-SCE) frame.
    base_cols = _numeric_feature_cols(
        train_df,
        target_col=target_col,
        time_col=time_col,
    )
    # SCE design = baseline numeric + context cols (present on both splits).
    sce_present = [c for c in sce_cols if c in train_enr.columns and c in test_enr.columns]
    sce_design_cols = list(dict.fromkeys([*base_cols, *sce_present]))

    y_train = train_df[target_col]
    y_test = test_df[target_col]

    baseline_model = Ridge(alpha=1.0, random_state=random_state)
    sce_model = Ridge(alpha=1.0, random_state=random_state)

    # Align baseline X to prepared train/test (numeric cols only).
    X_train_base = train_df.reindex(columns=base_cols)
    X_test_base = test_df.reindex(columns=base_cols)
    baseline_rmse, baseline_r2 = _fit_predict_rmse_r2(
        baseline_model, X_train_base, y_train, X_test_base, y_test
    )

    X_train_sce = train_enr.reindex(columns=sce_design_cols)
    X_test_sce = test_enr.reindex(columns=sce_design_cols)
    sce_rmse, sce_r2 = _fit_predict_rmse_r2(
        sce_model, X_train_sce, y_train, X_test_sce, y_test
    )

    return {
        "baseline_rmse": float(baseline_rmse),
        "baseline_r2": float(baseline_r2),
        "sce_rmse": float(sce_rmse),
        "sce_r2": float(sce_r2),
        "n_features_sce": int(len(sce_present)),
        "n_features_baseline": int(len(base_cols)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "use_cross_fitting": effective_cf,
    }


def _advantage_pct(baseline_rmse: float, sce_rmse: float) -> float:
    """SCE advantage in percent of baseline RMSE (positive = SCE better)."""
    if baseline_rmse == 0.0 or not np.isfinite(baseline_rmse):
        return 0.0
    return float(((baseline_rmse - sce_rmse) / baseline_rmse) * 100.0)


def run_permuted_target_equity(
    features: pd.DataFrame,
    *,
    target_col: str = "ret_1d",
    n_permutations: int = 5,
    seed: int = 42,
    hierarchy: EquityHierarchyConfig | None = None,
    sectors: pd.DataFrame | Path | str | None = None,
    use_cross_fitting: bool | None = None,
) -> dict[str, Any]:
    """Permuted-target leakage diagnostic on an equity features frame.

    Mirrors ``scripts.diagnostics.permuted_target.run_permuted_target``:
    compute real SCE advantage, then permute the target column ``n_permutations``
    times and require ``mean(permuted_advantage) < 1.0`` to pass.
    """
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}")

    real_metrics = evaluate_equity_sce(
        features,
        target_col=target_col,
        hierarchy=hierarchy,
        sectors=sectors,
        use_cross_fitting=use_cross_fitting,
        random_state=seed,
    )
    real_advantage = _advantage_pct(real_metrics["baseline_rmse"], real_metrics["sce_rmse"])

    rng = np.random.default_rng(seed)
    # Works on a copy of the column used after prepare aliasing; for equity the
    # source is typically ret_1d_log when ret_1d is absent — pinch both.
    work = features.copy()
    if target_col not in work.columns and target_col == "ret_1d" and "ret_1d_log" in work.columns:
        perm_source_col = "ret_1d_log"
    elif target_col in work.columns:
        perm_source_col = target_col
    else:
        raise ValueError(
            f"run_permuted_target_equity: target {target_col!r} missing from features "
            f"(and no ret_1d_log alias). columns={list(work.columns)[:20]}"
        )

    values = work[perm_source_col].to_numpy(copy=True)
    permuted_advantages: list[float] = []
    baseline_perm: list[float] = []
    sce_perm: list[float] = []
    for _ in range(n_permutations):
        perm_df = work.copy()
        perm_df[perm_source_col] = rng.permutation(values)
        # Keep ret_1d in sync if both exist so the enricher alias stays consistent.
        if perm_source_col == "ret_1d_log" and "ret_1d" in perm_df.columns:
            perm_df["ret_1d"] = perm_df["ret_1d_log"]
        elif perm_source_col == "ret_1d" and "ret_1d_log" in perm_df.columns:
            perm_df["ret_1d_log"] = perm_df["ret_1d"]
        metrics = evaluate_equity_sce(
            perm_df,
            target_col=target_col,
            hierarchy=hierarchy,
            sectors=sectors,
            use_cross_fitting=use_cross_fitting,
            random_state=seed,
        )
        baseline_perm.append(float(metrics["baseline_rmse"]))
        sce_perm.append(float(metrics["sce_rmse"]))
        permuted_advantages.append(
            _advantage_pct(metrics["baseline_rmse"], metrics["sce_rmse"])
        )

    perm_mean = float(np.mean(permuted_advantages)) if permuted_advantages else 0.0
    return {
        "baseline_rmse_real": float(real_metrics["baseline_rmse"]),
        "sce_rmse_real": float(real_metrics["sce_rmse"]),
        "baseline_rmse_permuted_mean": float(np.mean(baseline_perm)) if baseline_perm else 0.0,
        "sce_rmse_permuted_mean": float(np.mean(sce_perm)) if sce_perm else 0.0,
        "sce_advantage_real": float(real_advantage),
        "sce_advantage_permuted_mean": perm_mean,
        "permuted_advantages": permuted_advantages,
        "n_permutations": int(n_permutations),
        "pass": perm_mean < 1.0,
    }


def run_shuffled_groups_equity(
    features: pd.DataFrame,
    *,
    target_col: str = "ret_1d",
    n_permutations: int = 5,
    seed: int = 42,
    columns: Sequence[str] | None = None,
    mode: str = "all",
    hierarchy: EquityHierarchyConfig | None = None,
    sectors: pd.DataFrame | Path | str | None = None,
    use_cross_fitting: bool | None = None,
) -> dict[str, Any]:
    """Shuffled-groups structure diagnostic on an equity features frame.

    Mirrors ``scripts.diagnostics.shuffled_groups.run_shuffled_groups``.
    Default categorical columns come from the hierarchy knobs (``ticker``,
    ``sector``, ...). Note that ``sector`` / ``industry`` / ``mktcap_bucket``
    are joined inside the enricher from the static map — shuffling them on the
    *input* features frame only has effect when they are already present on
    ``features``; by default we therefore shuffle ``ticker`` (always present),
    which breaks the sector join key and is the equity-equivalent structure
    destroyer.
    """
    if mode not in {"all", "per-column"}:
        raise ValueError(f"Unsupported mode: {mode}")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1; got {n_permutations}")

    if hierarchy is None:
        hierarchy = EquityHierarchyConfig(target_col=target_col)

    categorical_cols = list(columns) if columns is not None else list(hierarchy.categorical_cols)
    # Only columns present on the input features can be shuffled pre-prepare.
    # ticker is always present; sector/* only if the caller pre-joined them.
    available_cols = [c for c in categorical_cols if c in features.columns]
    if not available_cols:
        # Fall back to ticker — the leaf group key always on build_features output.
        if "ticker" not in features.columns:
            raise ValueError(
                "No categorical columns available for shuffled-groups diagnostic "
                f"(looked for {categorical_cols}; features has {list(features.columns)[:20]})."
            )
        available_cols = ["ticker"]

    real_metrics = evaluate_equity_sce(
        features,
        target_col=target_col,
        hierarchy=hierarchy,
        sectors=sectors,
        use_cross_fitting=use_cross_fitting,
        random_state=seed,
    )
    real_advantage = _advantage_pct(real_metrics["baseline_rmse"], real_metrics["sce_rmse"])

    rng = np.random.default_rng(seed)
    shuffled_advantages: list[float] = []
    per_column_advantages: dict[str, list[float]] = {}

    def _eval_shuffled(df: pd.DataFrame) -> float:
        metrics = evaluate_equity_sce(
            df,
            target_col=target_col,
            hierarchy=hierarchy,
            sectors=sectors,
            use_cross_fitting=use_cross_fitting,
            random_state=seed,
        )
        return _advantage_pct(metrics["baseline_rmse"], metrics["sce_rmse"])

    if mode == "all":
        for _ in range(n_permutations):
            shuffled_df = features.copy()
            for col in available_cols:
                shuffled_df[col] = (
                    shuffled_df[col]
                    .sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
                    .to_numpy()
                )
            shuffled_advantages.append(float(_eval_shuffled(shuffled_df)))
    else:  # per-column
        for col in available_cols:
            col_advantages: list[float] = []
            for _ in range(n_permutations):
                shuffled_df = features.copy()
                shuffled_df[col] = (
                    shuffled_df[col]
                    .sample(frac=1.0, random_state=int(rng.integers(0, 1_000_000)))
                    .to_numpy()
                )
                col_advantages.append(float(_eval_shuffled(shuffled_df)))
            per_column_advantages[col] = col_advantages
            shuffled_advantages.extend(col_advantages)

    shuffled_mean = float(np.mean(shuffled_advantages)) if shuffled_advantages else 0.0
    threshold = 0.5 * real_advantage
    per_column_summary = {
        col: {
            "mean_advantage": float(np.mean(vals)) if vals else 0.0,
            "advantages": vals,
        }
        for col, vals in per_column_advantages.items()
    }
    return {
        "sce_advantage_real": float(real_advantage),
        "sce_advantage_shuffled_mean": shuffled_mean,
        "shuffled_advantages": shuffled_advantages,
        "columns_evaluated": available_cols,
        "per_column": per_column_summary,
        "mode": mode,
        "pass": (real_advantage - shuffled_mean) > threshold if real_advantage > 0 else False,
    }


def run_crossfit_ab_equity(
    features: pd.DataFrame,
    *,
    target_col: str = "ret_1d",
    hierarchy: EquityHierarchyConfig | None = None,
    sectors: pd.DataFrame | Path | str | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Cross-fit on/off A/B diagnostic on an equity features frame.

    Mirrors ``scripts.diagnostics.crossfit_ab.run_crossfit_ab``.
    ``leakage_signal_pp = (rmse_no_cf - rmse_cf) / rmse_no_cf * 100``.
    """
    with_cf = evaluate_equity_sce(
        features,
        target_col=target_col,
        hierarchy=hierarchy,
        sectors=sectors,
        use_cross_fitting=True,
        random_state=seed,
    )
    without_cf = evaluate_equity_sce(
        features,
        target_col=target_col,
        hierarchy=hierarchy,
        sectors=sectors,
        use_cross_fitting=False,
        random_state=seed,
    )

    rmse_cf = float(with_cf["sce_rmse"])
    rmse_no_cf = float(without_cf["sce_rmse"])
    if rmse_no_cf and np.isfinite(rmse_no_cf):
        leakage_signal_pp = float(((rmse_no_cf - rmse_cf) / rmse_no_cf) * 100.0)
    else:
        leakage_signal_pp = 0.0

    return {
        "rmse_cf": rmse_cf,
        "rmse_no_cf": rmse_no_cf,
        "r2_cf": float(with_cf.get("sce_r2", 0.0)),
        "r2_no_cf": float(without_cf.get("sce_r2", 0.0)),
        "baseline_rmse_cf": float(with_cf["baseline_rmse"]),
        "baseline_rmse_no_cf": float(without_cf["baseline_rmse"]),
        "leakage_signal_pp": leakage_signal_pp,
    }


def audit_feature_dominance_equity(
    importance_csv: str | Path,
    *,
    top_k: int = 3,
    threshold_pct: float = 70.0,
) -> dict[str, Any]:
    """Thin wrapper around :func:`scripts.run.audit_feature_dominance`.

    Does not duplicate the dominance math — reuses the shared helper so the
    equity and scripts diagnostic path stay in lockstep.
    """
    from scripts.run import audit_feature_dominance

    return audit_feature_dominance(
        Path(importance_csv),
        top_k=top_k,
        threshold_pct=threshold_pct,
    )


# Back-compat alias matching the brief's private-module name.
evaluate_config_dataframe_equity = evaluate_equity_sce
