"""
@module: equity.forecaster.run_smoke
@depends: pandas, equity.forecaster.*
@exports: run_smoke, main
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §7 / S5.5
@data_flow: prices [+ enriched features] -> ts-group split -> sector+residual+quantile
            -> predictions_h{N}.parquet + metadata.json

S5.5 single-fold train/test smoke runner. Uses a ts-group split (not full
walk-forward — that is S6). Entry point:

    python -m equity.forecaster.run_smoke \\
        --prices <prices.parquet> [--features <features.parquet>] \\
        [--sectors <sectors.csv>] [--output results/forecaster/smoke]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from equity.forecaster._splits import ts_group_split
from equity.forecaster.config import HorizonConfig, SmokeConfig
from equity.forecaster.metadata import (
    collect_git_sha,
    config_hash,
    write_metadata,
)
from equity.forecaster.quantile import QuantileHeadForecaster
from equity.forecaster.residual import InstrumentResidualForecaster
from equity.forecaster.sector_head import SectorHeadForecaster
from equity.forecaster.targets import add_forward_targets, forward_target_col

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "forecaster" / "smoke"


def _resolve_under_project_root(path: str | Path, project_root: Path | None = None) -> Path:
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


def run_smoke(
    prices: pd.DataFrame,
    *,
    features: Optional[pd.DataFrame] = None,
    sectors: Optional[pd.DataFrame] = None,
    config: Optional[SmokeConfig] = None,
    out_dir: str | Path | None = None,
    git_sha: Optional[str] = None,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    """Run a single ts-group train/test smoke of the two-layer + quantile heads.

    Parameters
    ----------
    prices:
        S1 prices panel (must carry ``ticker``, ``period_close_ts``, ``close``).
    features:
        Optional pre-built / SCE-enriched feature matrix. When None, a minimal
        feature block is derived from ``prices`` (close/volume + sector) so the
        smoke stays runnable without the full S3/S4 stack.
    sectors:
        Optional ``(ticker, sector[, industry, mktcap_bucket])`` frame.
    config:
        :class:`SmokeConfig` knobs.
    out_dir:
        Directory for ``predictions_h{N}.parquet`` + ``metadata.json``.
    git_sha / created_at:
        Overridable for tests; defaults from git / UTC now.

    Returns
    -------
    dict
        ``{out_dir, n_train, n_test, predictions, metadata, coverage}``.
    """
    cfg = config or SmokeConfig()
    hcfg = cfg.horizon
    horizons = tuple(hcfg.horizons)

    # 1. Forward targets from prices (labels only).
    priced = add_forward_targets(prices, horizons=horizons)

    # 2. Feature matrix. Always attach forward labels from `priced` (LABEL only —
    # make_design_matrix blocklists ret_h* so they never enter X).
    target_cols = [forward_target_col(h) for h in horizons]
    tcols = ["ticker", "period_close_ts"] + [
        c for c in target_cols if c in priced.columns
    ]

    if features is None:
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
        feat = priced[keep].copy()
    else:
        feat = features.copy()

    # Drop any pre-existing forward labels, then left-join the canonical ones
    # from prices so labels are always price-derived (never feature-derived).
    drop_existing = [c for c in target_cols if c in feat.columns]
    if drop_existing:
        feat = feat.drop(columns=drop_existing)
    feat = feat.merge(
        priced[tcols],
        on=["ticker", "period_close_ts"],
        how="left",
        sort=False,
    )

    feat = _ensure_sector(feat, sectors)

    # 3. ts-group split (P0 — never row-iloc).
    train, test = ts_group_split(
        feat, time_col=hcfg.time_col, test_frac=cfg.test_frac
    )

    # 4. Sector head on train (OOF for residual labels) + predict test.
    sector = SectorHeadForecaster(
        horizons=horizons,
        params=cfg.sector,
        horizon_cfg=hcfg,
        n_folds=hcfg.n_folds,
        seed=cfg.seed,
    )
    sector.fit(train, return_oof=True)
    sector_oof = sector.oof_predictions()
    sector_test = sector.predict(test)

    # 5. Residual head (OOF labels) + predict test.
    residual = InstrumentResidualForecaster(
        horizons=horizons,
        params=cfg.residual,
        horizon_cfg=hcfg,
        n_folds=hcfg.n_folds,
        seed=cfg.seed,
    )
    residual.fit(train, sector_preds=sector_oof, return_oof=True)
    residual_test = residual.predict(test, sector_preds=sector_test)

    # 6. Quantile heads.
    quantile = QuantileHeadForecaster(
        horizons=horizons,
        quantiles=hcfg.quantiles,
        params=cfg.quantile,
        horizon_cfg=hcfg,
        n_folds=hcfg.n_folds,
        seed=cfg.seed,
    )
    quantile.fit(train, return_oof=False)
    quantile_test = quantile.predict(test)

    # 7. Assemble per-horizon prediction frames for the test slice.
    predictions: dict[int, pd.DataFrame] = {}
    coverage: dict[int, float] = {}
    for h in horizons:
        y_col = forward_target_col(h)
        frame = pd.DataFrame(
            {
                "ticker": test["ticker"].to_numpy() if "ticker" in test.columns else None,
                "period_close_ts": test[hcfg.time_col].to_numpy(),
                y_col: test[y_col].to_numpy() if y_col in test.columns else np.nan,
                f"pred_sector_h{h}": residual_test[f"pred_sector_h{h}"].to_numpy(),
                f"pred_resid_h{h}": residual_test[f"pred_resid_h{h}"].to_numpy(),
                f"pred_h{h}": residual_test[f"pred_h{h}"].to_numpy(),
            }
        )
        for q in hcfg.quantiles:
            pct = int(round(float(q) * 100))
            qcol = f"pred_h{h}_q{pct:02d}"
            frame[qcol] = quantile_test[qcol].to_numpy()
        predictions[h] = frame

        # Empirical 90% coverage on the held-out fold (q05/q95).
        lo = frame.get(f"pred_h{h}_q05")
        hi = frame.get(f"pred_h{h}_q95")
        if lo is not None and hi is not None:
            coverage[h] = QuantileHeadForecaster.empirical_coverage(
                frame[y_col], lo, hi
            )

    # JSON-safe coverage: NaN/inf → None (json.dumps would emit invalid NaN).
    coverage = {
        h: (None if (v is None or not np.isfinite(v)) else float(v))
        for h, v in coverage.items()
    }

    # 8. Persist artefacts.
    dest = Path(out_dir) if out_dir is not None else DEFAULT_OUTPUT
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for h, pdf in predictions.items():
        path = dest / f"predictions_h{h}.parquet"
        pdf.to_parquet(path, index=False)
        written.append(str(path))

    sha = git_sha if git_sha is not None else collect_git_sha(PROJECT_ROOT)
    chash = config_hash(cfg)
    meta_path = write_metadata(
        dest,
        git_sha=sha,
        config_hash=chash,
        seed=cfg.seed,
        run_grade=cfg.run_grade,
        horizons=horizons,
        quantiles=hcfg.quantiles,
        created_at=created_at,
        extra={"n_train": int(len(train)), "n_test": int(len(test)), "coverage": coverage},
    )

    return {
        "out_dir": str(dest),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "predictions": predictions,
        "prediction_paths": written,
        "metadata_path": str(meta_path),
        "metadata": json.loads(meta_path.read_text(encoding="utf-8")),
        "coverage": coverage,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "S5.5 single-fold smoke: sector-head + residual + quantile heads. "
            "Writes predictions_h{N}.parquet and metadata.json."
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
        help="Optional pre-built / SCE-enriched features parquet.",
    )
    parser.add_argument(
        "--sectors",
        default=None,
        help="Optional sectors CSV/parquet (ticker, sector).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        help="Output directory under PROJECT_ROOT (default results/forecaster/smoke).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-frac", type=float, default=0.25)
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
    cfg = SmokeConfig(
        horizon=HorizonConfig(horizons=horizons, seed=args.seed),
        seed=args.seed,
        test_frac=args.test_frac,
        run_grade="exploratory",
    )
    out_dir = _resolve_under_project_root(args.output)
    result = run_smoke(
        prices,
        features=features,
        sectors=sectors,
        config=cfg,
        out_dir=out_dir,
    )
    summary = {
        "out_dir": result["out_dir"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "prediction_paths": result["prediction_paths"],
        "metadata_path": result["metadata_path"],
        "coverage": result["coverage"],
        "run_grade": result["metadata"].get("run_grade"),
    }
    print(json.dumps(summary, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
