"""
@module: equity.forecaster.baseline_compare
@depends: equity.forecaster.run_walk_forward, equity.metrics.accuracy
@exports: run_baseline_vs_sce, main
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §8 (S6.4 baseline-vs-SCE)
@data_flow: prices/features -> two walk-forward legs (sce on/off) -> comparison_report

S6.4 baseline-vs-SCE comparison. Both legs share the same WalkForwardConfig and
the same two-layer + quantile XGBoost/LightGBM stack; the only difference is
``sce_enrich``. Fold geometry is therefore deterministic and is asserted equal
across legs (timestamp bounds; ``n_features`` may differ because SCE adds cols).

Sign conventions (Δ = SCE − baseline), written into the markdown report:
  * Δ RMSE / Δ MAE  : **positive = SCE worse** (error went up)
  * Δ hit_rate      : **positive = SCE better**
  * Δ Sharpe/Sortino: **positive = SCE better**

Accuracy deltas come from the TEST-slice metrics in each leg's metadata;
Sharpe/Sortino deltas come from the VAL-slice portfolio metrics (S6.3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from equity.forecaster.config import WalkForwardConfig
from equity.forecaster.metadata import collect_git_sha, write_metadata
from equity.forecaster.run_walk_forward import (
    PROJECT_ROOT,
    _json_safe,
    _resolve_under_project_root,
    run_walk_forward,
)

DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "forecaster" / "baseline_vs_sce"


def _bound_key(b: Mapping[str, Any]) -> tuple:
    """Comparable fold-bound identity (geometry only — not n_features)."""
    return (
        int(b.get("fold_idx", -1)),
        str(b.get("train_start", "")),
        str(b.get("train_end", "")),
        str(b.get("val_start", "")),
        str(b.get("val_end", "")),
        str(b.get("test_start", "")),
        str(b.get("test_end", "")),
    )


def _assert_identical_fold_bounds(
    sce_bounds: Sequence[Mapping[str, Any]],
    base_bounds: Sequence[Mapping[str, Any]],
) -> None:
    if len(sce_bounds) != len(base_bounds):
        raise RuntimeError(
            f"fold count mismatch: sce={len(sce_bounds)} baseline={len(base_bounds)}"
        )
    for i, (sb, bb) in enumerate(zip(sce_bounds, base_bounds)):
        if _bound_key(sb) != _bound_key(bb):
            raise RuntimeError(
                f"fold_bounds geometry mismatch at fold {i}: "
                f"sce={_bound_key(sb)} baseline={_bound_key(bb)}"
            )


def _metric_num(m: Mapping[str, Any], key: str) -> float:
    v = m.get(key)
    if v is None:
        return float("nan")
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f if np.isfinite(f) else float("nan")


def _delta(a: float, b: float) -> Optional[float]:
    """a − b as JSON-safe Optional[float] (None when non-finite)."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return None
    return float(a - b)


def build_comparison(
    sce_meta: Mapping[str, Any],
    base_meta: Mapping[str, Any],
    *,
    horizons: Sequence[int],
) -> Dict[str, Any]:
    """Assemble the machine-readable comparison payload from two leg metadatas."""
    sce_metrics = sce_meta.get("metrics") or {}
    base_metrics = base_meta.get("metrics") or {}
    per_h: Dict[str, Dict[str, Any]] = {}
    for h in horizons:
        hk = str(int(h))
        sm = sce_metrics.get(hk) or {}
        bm = base_metrics.get(hk) or {}
        per_h[hk] = {
            "sce": {
                "rmse_mean": sm.get("rmse_mean"),
                "mae_mean": sm.get("mae_mean"),
                "hit_rate_mean": sm.get("hit_rate_mean"),
                "sharpe": sm.get("sharpe"),
                "sortino": sm.get("sortino"),
            },
            "baseline": {
                "rmse_mean": bm.get("rmse_mean"),
                "mae_mean": bm.get("mae_mean"),
                "hit_rate_mean": bm.get("hit_rate_mean"),
                "sharpe": bm.get("sharpe"),
                "sortino": bm.get("sortino"),
            },
            "delta": {
                # positive Δ RMSE/MAE = SCE worse; positive Δ hit/sharpe/sortino = SCE better
                "d_rmse": _delta(_metric_num(sm, "rmse_mean"), _metric_num(bm, "rmse_mean")),
                "d_mae": _delta(_metric_num(sm, "mae_mean"), _metric_num(bm, "mae_mean")),
                "d_hit_rate": _delta(
                    _metric_num(sm, "hit_rate_mean"), _metric_num(bm, "hit_rate_mean")
                ),
                "d_sharpe": _delta(_metric_num(sm, "sharpe"), _metric_num(bm, "sharpe")),
                "d_sortino": _delta(_metric_num(sm, "sortino"), _metric_num(bm, "sortino")),
            },
        }

    return {
        "horizons": [int(h) for h in horizons],
        "chosen_horizon": {
            "sce": sce_meta.get("chosen_horizon"),
            "baseline": base_meta.get("chosen_horizon"),
        },
        "sign_conventions": {
            "d_rmse": "SCE - baseline; positive = SCE worse",
            "d_mae": "SCE - baseline; positive = SCE worse",
            "d_hit_rate": "SCE - baseline; positive = SCE better",
            "d_sharpe": "SCE - baseline; positive = SCE better (VAL decile L/S)",
            "d_sortino": "SCE - baseline; positive = SCE better (VAL decile L/S)",
            "accuracy_slice": "TEST",
            "portfolio_slice": "VAL",
        },
        "per_horizon": per_h,
        "walk_forward": {
            "sce": sce_meta.get("walk_forward"),
            "baseline": base_meta.get("walk_forward"),
        },
    }


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        return "n/a"
    return f"{f:.6g}"


def render_comparison_md(report: Mapping[str, Any]) -> str:
    """Human-readable markdown summary table for the comparison report."""
    lines: List[str] = [
        "# Baseline vs SCE comparison",
        "",
        "Both legs share the same `WalkForwardConfig` and the same two-layer + "
        "quantile forecaster stack. The **only** difference is `sce_enrich` "
        "(SCE context enrichment on vs off).",
        "",
        "## Sign conventions",
        "",
        "| Δ metric | Formula | Positive means | Slice |",
        "|---|---|---|---|",
        "| d_rmse | SCE − baseline | SCE **worse** | TEST |",
        "| d_mae | SCE − baseline | SCE **worse** | TEST |",
        "| d_hit_rate | SCE − baseline | SCE **better** | TEST |",
        "| d_sharpe | SCE − baseline | SCE **better** | VAL (decile L/S) |",
        "| d_sortino | SCE − baseline | SCE **better** | VAL (decile L/S) |",
        "",
        "## Chosen horizon (VAL criterion = Sharpe)",
        "",
        f"- SCE: `{report.get('chosen_horizon', {}).get('sce')}`",
        f"- Baseline: `{report.get('chosen_horizon', {}).get('baseline')}`",
        "",
        "## Per-horizon deltas",
        "",
        "| h | d_rmse | d_mae | d_hit_rate | d_sharpe | d_sortino | "
        "sce_rmse | base_rmse | sce_sharpe | base_sharpe |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    per_h = report.get("per_horizon") or {}
    for hk in sorted(per_h.keys(), key=lambda x: int(x)):
        block = per_h[hk]
        d = block.get("delta") or {}
        s = block.get("sce") or {}
        b = block.get("baseline") or {}
        lines.append(
            f"| {hk} | {_fmt(d.get('d_rmse'))} | {_fmt(d.get('d_mae'))} | "
            f"{_fmt(d.get('d_hit_rate'))} | {_fmt(d.get('d_sharpe'))} | "
            f"{_fmt(d.get('d_sortino'))} | {_fmt(s.get('rmse_mean'))} | "
            f"{_fmt(b.get('rmse_mean'))} | {_fmt(s.get('sharpe'))} | "
            f"{_fmt(b.get('sharpe'))} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _baseline_vs_sce_extra(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact per-horizon Δ block for canonical SCE metadata.extra."""
    out: Dict[str, Any] = {}
    for hk, block in (report.get("per_horizon") or {}).items():
        d = block.get("delta") or {}
        out[str(hk)] = {
            "d_rmse": d.get("d_rmse"),
            "d_mae": d.get("d_mae"),
            "d_hit_rate": d.get("d_hit_rate"),
            "d_sharpe": d.get("d_sharpe"),
            "d_sortino": d.get("d_sortino"),
        }
    return out


def run_baseline_vs_sce(
    prices: pd.DataFrame,
    features: Optional[pd.DataFrame] = None,
    sectors: Optional[pd.DataFrame] = None,
    cfg: Optional[WalkForwardConfig] = None,
    *,
    out_dir: str | Path | None = None,
    git_sha: Optional[str] = None,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    """Run SCE and baseline walk-forward legs and write a comparison report.

    Parameters
    ----------
    prices / features / sectors / cfg:
        Forwarded to both :func:`run_walk_forward` legs unchanged (same folds).
    out_dir:
        Root output directory. Legs land under ``legs/sce`` and ``legs/baseline``;
        comparison reports land at the root.

    Returns
    -------
    dict
        ``{out_dir, sce, baseline, comparison, comparison_report_json,
        comparison_report_md, metadata_path, metadata}``.
    """
    cfg = cfg or WalkForwardConfig()
    horizons = tuple(int(h) for h in cfg.horizons)
    dest = Path(out_dir) if out_dir is not None else DEFAULT_OUTPUT
    dest.mkdir(parents=True, exist_ok=True)
    legs_dir = dest / "legs"
    sce_dir = legs_dir / "sce"
    base_dir = legs_dir / "baseline"

    sha = git_sha if git_sha is not None else collect_git_sha(PROJECT_ROOT)

    sce_result = run_walk_forward(
        prices,
        features=features,
        sectors=sectors,
        cfg=cfg,
        out_dir=sce_dir,
        sce_enrich=True,
        git_sha=sha,
        created_at=created_at,
    )
    base_result = run_walk_forward(
        prices,
        features=features,
        sectors=sectors,
        cfg=cfg,
        out_dir=base_dir,
        sce_enrich=False,
        git_sha=sha,
        created_at=created_at,
    )

    sce_meta = sce_result["metadata"]
    base_meta = base_result["metadata"]
    sce_bounds = (sce_meta.get("walk_forward") or {}).get("fold_bounds") or []
    base_bounds = (base_meta.get("walk_forward") or {}).get("fold_bounds") or []
    _assert_identical_fold_bounds(sce_bounds, base_bounds)

    comparison = build_comparison(sce_meta, base_meta, horizons=horizons)
    comparison = _json_safe(comparison)

    json_path = dest / "comparison_report.json"
    md_path = dest / "comparison_report.md"
    json_path.write_text(
        json.dumps(comparison, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_comparison_md(comparison), encoding="utf-8")

    # Enrich the SCE leg's metadata with the full canonical schema (in place on disk).
    # SCE leg metadata is the run-level canonical artifact.
    bvs = _baseline_vs_sce_extra(comparison)
    bvs_block = {
        **bvs,
        "comparison_report_path": str(json_path),
        "comparison_report_md_path": str(md_path),
        "baseline_metadata_path": base_result["metadata_path"],
        "chosen_horizon_baseline": base_meta.get("chosen_horizon"),
    }
    # Re-write SCE metadata with baseline_vs_sce attached + comparison path.
    # Preserve existing walk_forward / metrics / chosen_horizon.
    extra = {
        "walk_forward": sce_meta.get("walk_forward"),
        "metrics": sce_meta.get("metrics"),
        "chosen_horizon": sce_meta.get("chosen_horizon"),
        "baseline_vs_sce": _json_safe(bvs_block),
    }
    # Also drop a root-level canonical metadata.json pointing at the SCE leg.
    from equity.forecaster.metadata import config_hash

    root_meta_path = write_metadata(
        dest,
        git_sha=str(sce_meta.get("git_sha") or sha),
        config_hash=str(sce_meta.get("config_hash") or config_hash(cfg)),
        seed=int(sce_meta.get("seed", cfg.seed)),
        run_grade=str(sce_meta.get("run_grade", cfg.run_grade)),
        horizons=horizons,
        quantiles=tuple(float(q) for q in cfg.quantiles),
        created_at=created_at or sce_meta.get("created_at"),
        extra=extra,
    )
    # Patch the SCE leg metadata.json so legs/sce is also complete.
    write_metadata(
        sce_dir,
        git_sha=str(sce_meta.get("git_sha") or sha),
        config_hash=str(sce_meta.get("config_hash") or config_hash(cfg)),
        seed=int(sce_meta.get("seed", cfg.seed)),
        run_grade=str(sce_meta.get("run_grade", cfg.run_grade)),
        horizons=horizons,
        quantiles=tuple(float(q) for q in cfg.quantiles),
        created_at=created_at or sce_meta.get("created_at"),
        extra=extra,
    )

    root_meta = json.loads(root_meta_path.read_text(encoding="utf-8"))
    # Keep in-memory SCE result metadata in sync with disk.
    sce_result["metadata"] = json.loads((sce_dir / "metadata.json").read_text(encoding="utf-8"))

    return {
        "out_dir": str(dest),
        "sce": sce_result,
        "baseline": base_result,
        "comparison": comparison,
        "comparison_report_json": str(json_path),
        "comparison_report_md": str(md_path),
        "metadata_path": str(root_meta_path),
        "metadata": root_meta,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "S6.4 baseline-vs-SCE: run two walk-forward legs (sce_enrich on/off) "
            "and write comparison_report.{json,md} + canonical metadata.json."
        )
    )
    parser.add_argument("--prices", required=True, help="S1 prices parquet.")
    parser.add_argument("--features", default=None, help="Optional pre-SCE features parquet.")
    parser.add_argument("--sectors", default=None, help="Optional sectors CSV/parquet.")
    parser.add_argument("--output", required=True, help="Output root under PROJECT_ROOT.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-grade", default="diagnostic")
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
    result = run_baseline_vs_sce(
        prices,
        features=features,
        sectors=sectors,
        cfg=cfg,
        out_dir=out_dir,
    )
    summary = {
        "out_dir": result["out_dir"],
        "metadata_path": result["metadata_path"],
        "comparison_report_json": result["comparison_report_json"],
        "comparison_report_md": result["comparison_report_md"],
        "chosen_horizon": {
            "sce": result["metadata"].get("chosen_horizon"),
            "baseline": result["baseline"]["metadata"].get("chosen_horizon"),
        },
        "baseline_vs_sce": result["metadata"].get("baseline_vs_sce"),
        "n_folds_sce": result["sce"]["n_folds"],
        "n_folds_baseline": result["baseline"]["n_folds"],
    }
    print(json.dumps(summary, indent=2, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
