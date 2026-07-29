"""
@module: equity.forecaster.run_walk_forward
@depends: pandas, equity.forecaster.*, equity.metrics.accuracy, equity.sce.enrich
@exports: run_walk_forward, main
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §8 / S6.1 + S6.2
@data_flow: prices [+ features] -> walk-forward folds -> per-fold SCE refit
            -> sector+residual+quantile -> predictions_h{N}.parquet + metadata.json

S6.1 walk-forward backtest runner + S6.2 per-horizon accuracy metrics. Per fold
the pipeline mirrors ``run_smoke`` (targets after SCE, price-derived labels) but
uses a sliding train/val/test geometry and optionally re-fits
:class:`EquityContextEnricher` PIT-safely on the train slice only.

Artifact layout under ``out_dir``::

    fold_{k:02d}/predictions_h{N}.parquet   # val+test rows, ``split`` column tags slice
    metadata.json                           # walk_forward + metrics + mono gate

Entry point::

    python -m equity.forecaster.run_walk_forward \\
        --prices <prices.parquet> [--features <features.parquet>] \\
        [--sectors <sectors.csv>] --output results/forecaster/walk_forward \\
        [--train-window 1260 --val-window 63 --test-window 63 --step 63]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from equity.diagnostics.walk_forward_monotonicity import run_walk_forward_monotonicity
from equity.forecaster._splits import WalkForwardFold, walk_forward_folds
from equity.forecaster.config import WalkForwardConfig
from equity.forecaster.metadata import collect_git_sha, config_hash, write_metadata
from equity.forecaster.quantile import QuantileHeadForecaster
from equity.forecaster.residual import InstrumentResidualForecaster
from equity.forecaster.sector_head import SectorHeadForecaster
from equity.forecaster.targets import add_forward_targets, forward_target_col
from equity.metrics.accuracy import aggregate_accuracy

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "forecaster" / "walk_forward"


def _resolve_under_project_root(path: str | Path, project_root: Path | None = None) -> Path:
    """Contain ``path`` under the project root. Mirrors ``run_smoke``."""
    root = project_root if project_root is not None else PROJECT_ROOT
    raw = Path(path)
    if ".." in list(raw.parts):
        raise ValueError(f"Refusing path with '..': {path}")
    resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Refusing path outside project root ({root}): {path}") from exc
    return resolved


def _ensure_sector(frame: pd.DataFrame, sectors: Optional[pd.DataFrame]) -> pd.DataFrame:
    out = frame.copy()
    if "sector" in out.columns:
        out["sector"] = out["sector"].fillna("unknown").astype(str)
        return out
    if sectors is None:
        out["sector"] = "unknown"
        return out
    sec = sectors.copy()
    if "ticker" not in sec.columns or "sector" not in sec.columns:
        raise ValueError("sectors frame must have columns 'ticker' and 'sector'")
    out = out.merge(sec[["ticker", "sector"]], on="ticker", how="left", sort=False)
    out["sector"] = out["sector"].fillna("unknown").astype(str)
    return out


def _slice_by_bounds(
    frame: pd.DataFrame,
    *,
    time_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Inclusive timestamp slice on ``time_col`` (never mid torn for equal bounds)."""
    ts = pd.to_datetime(frame[time_col])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    # Align tz so comparisons stay valid across tz-aware / tz-naive mixes.
    if getattr(ts.dt, "tz", None) is not None:
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(ts.dt.tz)
        else:
            start_ts = start_ts.tz_convert(ts.dt.tz)
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize(ts.dt.tz)
        else:
            end_ts = end_ts.tz_convert(ts.dt.tz)
    mask = (ts >= start_ts) & (ts <= end_ts)
    return frame.loc[mask].copy()


def _minimal_features_from_prices(priced: pd.DataFrame) -> pd.DataFrame:
    keep = [
        c
        for c in (
            "ticker",
            "period_close_ts",
            "close",
            "open",
            "high",
            "low",
            "volume",
            "adj_close",
        )
        if c in priced.columns
    ]
    return priced[keep].copy()


def _attach_labels(
    feat: pd.DataFrame,
    priced: pd.DataFrame,
    horizons: Sequence[int],
    *,
    time_col: str,
    ticker_col: str,
) -> pd.DataFrame:
    """Left-join price-derived ``ret_hN`` labels onto a feature frame."""
    target_cols = [forward_target_col(h) for h in horizons]
    tcols = [ticker_col, time_col] + [c for c in target_cols if c in priced.columns]
    out = feat.copy()
    drop_existing = [c for c in target_cols if c in out.columns]
    if drop_existing:
        out = out.drop(columns=drop_existing)
    return out.merge(
        priced[tcols],
        on=[ticker_col, time_col],
        how="left",
        sort=False,
    )


def _assemble_pred_frame(
    slice_df: pd.DataFrame,
    *,
    residual_preds: pd.DataFrame,
    quantile_preds: pd.DataFrame,
    horizons: Sequence[int],
    quantiles: Sequence[float],
    time_col: str,
    ticker_col: str,
    split: str,
) -> Dict[int, pd.DataFrame]:
    """Build per-horizon prediction frames for one val/test slice."""
    out: Dict[int, pd.DataFrame] = {}
    n = len(slice_df)
    for h in horizons:
        y_col = forward_target_col(h)
        frame = pd.DataFrame(
            {
                ticker_col: (
                    slice_df[ticker_col].to_numpy()
                    if ticker_col in slice_df.columns
                    else np.array([None] * n)
                ),
                time_col: slice_df[time_col].to_numpy(),
                "split": split,
                y_col: (
                    slice_df[y_col].to_numpy() if y_col in slice_df.columns else np.full(n, np.nan)
                ),
                f"pred_sector_h{h}": residual_preds[f"pred_sector_h{h}"].to_numpy(),
                f"pred_resid_h{h}": residual_preds[f"pred_resid_h{h}"].to_numpy(),
                f"pred_h{h}": residual_preds[f"pred_h{h}"].to_numpy(),
            }
        )
        for q in quantiles:
            pct = int(round(float(q) * 100))
            qcol = f"pred_h{h}_q{pct:02d}"
            frame[qcol] = quantile_preds[qcol].to_numpy()
        out[int(h)] = frame
    return out


def _json_safe(obj: Any) -> Any:
    """Recursively coerce NaN/inf -> None and Timestamps -> str for JSON."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is None:
        return None
    return obj


def run_walk_forward(
    prices: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
    sectors: Optional[pd.DataFrame] = None,
    cfg: Optional[WalkForwardConfig] = None,
    *,
    out_dir: str | Path | None = None,
    sce_enrich: bool = True,
    git_sha: Optional[str] = None,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    """Run multi-fold walk-forward of the two-layer + quantile heads (S6.1/S6.2).

    Parameters
    ----------
    prices:
        S1 prices panel (``ticker``, ``period_close_ts``, ``close``).
    features:
        Optional pre-built feature matrix (S3). When ``sce_enrich=True`` this is
        the *pre-SCE* frame that is re-fit per fold. When ``sce_enrich=False``
        this is used as-is (or a minimal OHLCV block is derived from ``prices``).
    sectors:
        Optional ``(ticker, sector[, industry, mktcap_bucket])`` frame.
    cfg:
        :class:`WalkForwardConfig` knobs.
    out_dir:
        Directory for per-fold ``predictions_h{N}.parquet`` + top-level
        ``metadata.json``.
    sce_enrich:
        When True, re-fit :class:`EquityContextEnricher` on each fold's train
        slice and transform val/test (PIT-safe). Requires usable feature inputs
        (passed ``features`` or built via ``build_features``).
    git_sha / created_at:
        Overridable for tests; defaults from git / UTC now.

    Returns
    -------
    dict
        ``{out_dir, n_folds, folds, fold_predictions_val, fold_predictions_test,
        metrics, metadata, metadata_path, prediction_paths, monotonicity}``.
    """
    cfg = cfg or WalkForwardConfig()
    horizons = tuple(int(h) for h in cfg.horizons)
    quantiles = tuple(float(q) for q in cfg.quantiles)
    hcfg = cfg.to_horizon_config()
    time_col = cfg.time_col
    ticker_col = cfg.ticker_col

    # Labels always from prices (price-derived). Merged AFTER any SCE transform.
    priced = add_forward_targets(prices, horizons=horizons, time_col=time_col, ticker_col=ticker_col)

    if features is None:
        if sce_enrich:
            from equity.features.build import build_features

            base_features = build_features(prices)
        else:
            base_features = _minimal_features_from_prices(priced)
    else:
        base_features = features.copy()

    base_features = _ensure_sector(base_features, sectors)

    # Fold geometry on the (unique timestamp) axis of the feature panel.
    folds = walk_forward_folds(base_features, cfg, time_col=time_col)

    mono_input = [f.to_monotonicity_record() for f in folds]
    mono = run_walk_forward_monotonicity(mono_input, strict=False)
    if not mono.get("pass", False):
        raise RuntimeError(
            "walk-forward monotonicity gate FAILED: "
            f"n_violations={mono.get('n_violations')} violations={mono.get('violations')}"
        )

    dest = Path(out_dir) if out_dir is not None else DEFAULT_OUTPUT
    dest.mkdir(parents=True, exist_ok=True)

    fold_preds_val: List[Dict[int, pd.DataFrame]] = []
    fold_preds_test: List[Dict[int, pd.DataFrame]] = []
    fold_bounds_meta: List[Dict[str, Any]] = []
    written: List[str] = []

    for fold_idx, fold in enumerate(folds):
        train_raw = _slice_by_bounds(
            base_features, time_col=time_col, start=fold.train_start, end=fold.train_end
        )
        val_raw = _slice_by_bounds(
            base_features, time_col=time_col, start=fold.val_start, end=fold.val_end
        )
        test_raw = _slice_by_bounds(
            base_features, time_col=time_col, start=fold.test_start, end=fold.test_end
        )
        if train_raw.empty or val_raw.empty or test_raw.empty:
            raise RuntimeError(
                f"fold {fold_idx}: empty slice after bound cut "
                f"(train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)})"
            )

        if sce_enrich:
            from equity.sce import EquityContextEnricher, EquityHierarchyConfig

            # PIT-safe per-fold refit (mirrors sce_reuse.evaluate_equity_sce).
            # EquityContextEnricher._prepare drops ret_h* — labels join AFTER transform.
            # Public surface stores the fitted engine on ``_engine`` (no ``.engine``
            # property); use that + ``engine.transform`` on val/test.
            enricher = EquityContextEnricher(
                hierarchy=EquityHierarchyConfig(),
                sectors=sectors if sectors is not None else None,
            )
            train_feat = enricher.fit_transform(train_raw)
            if enricher._engine is None:
                raise RuntimeError("EquityContextEnricher.fit_transform did not set _engine")
            # transform expects columns already prepared like fit — re-prepare
            # val/test through the same pipeline (tz, hierarchy join, ret_1d alias)
            # then run the fitted engine's transform (in-sample stats from train).
            val_prep = enricher._prepare(val_raw)
            test_prep = enricher._prepare(test_raw)
            val_feat = enricher._engine.transform(val_prep)
            test_feat = enricher._engine.transform(test_prep)
            val_feat = enricher._post_filter_interactions(val_feat)
            test_feat = enricher._post_filter_interactions(test_feat)
        else:
            train_feat, val_feat, test_feat = train_raw, val_raw, test_raw

        train = _attach_labels(
            train_feat, priced, horizons, time_col=time_col, ticker_col=ticker_col
        )
        val = _attach_labels(
            val_feat, priced, horizons, time_col=time_col, ticker_col=ticker_col
        )
        test = _attach_labels(
            test_feat, priced, horizons, time_col=time_col, ticker_col=ticker_col
        )
        train = _ensure_sector(train, sectors)
        val = _ensure_sector(val, sectors)
        test = _ensure_sector(test, sectors)

        # Sector head: OOF on train for residual labels; full-refit predict on val/test.
        sector = SectorHeadForecaster(
            horizons=horizons,
            params=cfg.sector,
            horizon_cfg=hcfg,
            n_folds=cfg.n_folds,
            seed=cfg.seed,
        )
        sector.fit(train, return_oof=True)
        sector_oof = sector.oof_predictions()
        sector_val = sector.predict(val)
        sector_test = sector.predict(test)

        residual = InstrumentResidualForecaster(
            horizons=horizons,
            params=cfg.residual,
            horizon_cfg=hcfg,
            n_folds=cfg.n_folds,
            seed=cfg.seed,
        )
        residual.fit(train, sector_preds=sector_oof, return_oof=True)
        residual_val = residual.predict(val, sector_preds=sector_val)
        residual_test = residual.predict(test, sector_preds=sector_test)

        quantile = QuantileHeadForecaster(
            horizons=horizons,
            quantiles=quantiles,
            params=cfg.quantile,
            horizon_cfg=hcfg,
            n_folds=cfg.n_folds,
            seed=cfg.seed,
        )
        quantile.fit(train, return_oof=False)
        quantile_val = quantile.predict(val)
        quantile_test = quantile.predict(test)

        val_preds = _assemble_pred_frame(
            val,
            residual_preds=residual_val,
            quantile_preds=quantile_val,
            horizons=horizons,
            quantiles=quantiles,
            time_col=time_col,
            ticker_col=ticker_col,
            split="val",
        )
        test_preds = _assemble_pred_frame(
            test,
            residual_preds=residual_test,
            quantile_preds=quantile_test,
            horizons=horizons,
            quantiles=quantiles,
            time_col=time_col,
            ticker_col=ticker_col,
            split="test",
        )
        fold_preds_val.append(val_preds)
        fold_preds_test.append(test_preds)

        fold_dir = dest / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        for h in horizons:
            combined = pd.concat([val_preds[h], test_preds[h]], ignore_index=True)
            path = fold_dir / f"predictions_h{h}.parquet"
            combined.to_parquet(path, index=False)
            written.append(str(path))

        fold_bounds_meta.append(
            {
                "fold_idx": int(fold_idx),
                "train_start": fold.train_start.isoformat(),
                "train_end": fold.train_end.isoformat(),
                "val_start": fold.val_start.isoformat(),
                "val_end": fold.val_end.isoformat(),
                "test_start": fold.test_start.isoformat(),
                "test_end": fold.test_end.isoformat(),
                "n_train": int(len(train)),
                "n_val": int(len(val)),
                "n_test": int(len(test)),
            }
        )

    metrics_raw = aggregate_accuracy(fold_preds_test, horizons)
    # JSON keys must be strings; values NaN-safe.
    metrics_meta = {
        str(h): _json_safe(vals) for h, vals in metrics_raw.items()
    }

    mono_meta = _json_safe(mono)
    sha = git_sha if git_sha is not None else collect_git_sha(PROJECT_ROOT)
    chash = config_hash(cfg)
    extra = {
        "walk_forward": {
            "n_folds": int(len(folds)),
            "train_window": int(cfg.train_window),
            "val_window": int(cfg.val_window),
            "test_window": int(cfg.test_window),
            "step": int(cfg.step),
            "sce_enrich": bool(sce_enrich),
            "fold_bounds": fold_bounds_meta,
            "monotonicity": mono_meta,
        },
        "metrics": metrics_meta,
    }
    meta_path = write_metadata(
        dest,
        git_sha=sha,
        config_hash=chash,
        seed=cfg.seed,
        run_grade=cfg.run_grade,
        horizons=horizons,
        quantiles=quantiles,
        created_at=created_at,
        extra=extra,
    )

    return {
        "out_dir": str(dest),
        "n_folds": int(len(folds)),
        "folds": folds,
        "fold_predictions_val": fold_preds_val,
        "fold_predictions_test": fold_preds_test,
        "metrics": metrics_raw,
        "monotonicity": mono,
        "prediction_paths": written,
        "metadata_path": str(meta_path),
        "metadata": json.loads(meta_path.read_text(encoding="utf-8")),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "S6.1 walk-forward: sector-head + residual + quantile heads per fold. "
            "Writes fold_XX/predictions_h{N}.parquet and metadata.json "
            "(includes S6.2 per-horizon accuracy metrics)."
        )
    )
    parser.add_argument(
        "--prices",
        required=True,
        help="Path to S1 prices parquet (ticker, period_close_ts, close, ...).",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Optional pre-built (pre-SCE) features parquet.",
    )
    parser.add_argument(
        "--sectors",
        default=None,
        help="Optional sectors CSV/parquet (ticker, sector).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory under PROJECT_ROOT.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-grade", default="diagnostic")
    parser.add_argument(
        "--no-sce",
        action="store_true",
        help="Skip per-fold SCE enrich (use raw features / OHLCV only).",
    )
    parser.add_argument("--train-window", type=int, default=1260)
    parser.add_argument("--val-window", type=int, default=63)
    parser.add_argument("--test-window", type=int, default=63)
    parser.add_argument("--step", type=int, default=63)
    parser.add_argument(
        "--horizons",
        default="1,5,10,21,63",
        help="Comma-separated horizons (default 1,5,10,21,63).",
    )
    args = parser.parse_args(argv)

    prices = pd.read_parquet(args.prices)
    features = pd.read_parquet(args.features) if args.features else None
    sectors = None
    if args.sectors:
        sp = Path(args.sectors)
        sectors = pd.read_csv(sp) if sp.suffix.lower() == ".csv" else pd.read_parquet(sp)

    horizons = tuple(int(x) for x in args.horizons.split(",") if x.strip())
    cfg = WalkForwardConfig(
        train_window=int(args.train_window),
        val_window=int(args.val_window),
        test_window=int(args.test_window),
        step=int(args.step),
        horizons=horizons,
        seed=int(args.seed),
        run_grade=str(args.run_grade),
    )
    out_dir = _resolve_under_project_root(args.output)
    result = run_walk_forward(
        prices,
        features=features,
        sectors=sectors,
        cfg=cfg,
        out_dir=out_dir,
        sce_enrich=not bool(args.no_sce),
    )
    summary = {
        "out_dir": result["out_dir"],
        "n_folds": result["n_folds"],
        "prediction_paths": result["prediction_paths"],
        "metadata_path": result["metadata_path"],
        "run_grade": result["metadata"].get("run_grade"),
        "metrics": result["metadata"].get("metrics"),
        "walk_forward": {
            k: result["metadata"]["walk_forward"][k]
            for k in (
                "n_folds",
                "train_window",
                "val_window",
                "test_window",
                "step",
                "sce_enrich",
            )
            if k in result["metadata"].get("walk_forward", {})
        },
        "monotonicity_pass": result["monotonicity"].get("pass"),
    }
    print(json.dumps(summary, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
