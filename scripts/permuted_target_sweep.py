"""
Grid-search sweep over SCE config parameters, measured by permuted-target leakage signal.

The sweep patches the TOML config in-memory for each combination,
runs N permutations, and records the mean permuted advantage.
Goal: find which parameter settings bring the permuted mean advantage
below the pass_threshold (default 5.0 pp).

Usage:
    python -m scripts.permuted_target_sweep --dataset rental_poland_short
    python -m scripts.permuted_target_sweep --dataset rental_poland_short --n-permutations 20 --pass-threshold 5.0
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.diagnostics._common import evaluate_config_dataframe, load_config_and_dataset

RESULTS_DIR = PROJECT_ROOT / "results"
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Grid definition — values to sweep for each parameter
# ---------------------------------------------------------------------------
GRID: dict[str, list[Any]] = {
    "min_group_size":       [3, 5, 10],
    "include_interactions": [True, False],
    "include_fold_variance":[True, False],
    "missing_threshold":    [0.2, 0.1],
}

# Parameters that live in [sce] section vs [run.feature_pruning]
SCE_PARAMS = {"min_group_size", "include_interactions", "include_fold_variance"}
PRUNING_PARAMS = {"missing_threshold"}


def _patch_config(base_config: dict, combo: dict[str, Any]) -> dict:
    """Return a deep-copied config with the combo values applied."""
    cfg = copy.deepcopy(base_config)
    cfg.setdefault("sce", {})
    cfg.setdefault("run", {})
    cfg["run"].setdefault("feature_pruning", {})

    for key, val in combo.items():
        if key in SCE_PARAMS:
            cfg["sce"][key] = val
        elif key in PRUNING_PARAMS:
            cfg["run"]["feature_pruning"][key] = val
    return cfg


def _run_combo(
    config: dict,
    config_name: str,
    df,
    target_col: str,
    combo: dict[str, Any],
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    patched = _patch_config(config, combo)
    rng = np.random.default_rng(seed)
    values = df[target_col].to_numpy(copy=True)

    real_metrics = evaluate_config_dataframe(patched, config_name, df, target_col)
    real_advantage = (
        (real_metrics["baseline_rmse"] - real_metrics["sce_rmse"])
        / real_metrics["baseline_rmse"]
    ) * 100

    permuted_advantages: list[float] = []
    for _ in range(n_permutations):
        perm_df = df.copy()
        perm_df[target_col] = rng.permutation(values)
        m = evaluate_config_dataframe(patched, config_name, perm_df, target_col)
        adv = ((m["baseline_rmse"] - m["sce_rmse"]) / m["baseline_rmse"]) * 100
        permuted_advantages.append(float(adv))

    return {
        **combo,
        "real_advantage_pct": round(float(real_advantage), 3),
        "baseline_rmse": round(float(real_metrics["baseline_rmse"]), 2),
        "sce_rmse": round(float(real_metrics["sce_rmse"]), 2),
        "permuted_mean_pp": round(float(np.mean(permuted_advantages)), 3),
        "permuted_std_pp": round(float(np.std(permuted_advantages)), 3),
        "permuted_advantages": permuted_advantages,
    }


def run_sweep(
    dataset: str,
    grid: dict[str, list[Any]],
    n_permutations: int = 10,
    seed: int = 42,
    pass_threshold: float = 5.0,
) -> list[dict[str, Any]]:
    config, _, df, target_col = load_config_and_dataset(dataset)

    keys = list(grid.keys())
    values_list = [grid[k] for k in keys]
    combos = [dict(zip(keys, v)) for v in itertools.product(*values_list)]
    total = len(combos)

    print(f"\n=== Permuted-target sweep: {dataset} ===")
    print(f"Grid: {total} combinations × {n_permutations} permutations = {total * n_permutations} model fits")
    print(f"Pass threshold: permuted_mean_pp < {pass_threshold}\n")

    results: list[dict[str, Any]] = []
    for i, combo in enumerate(combos, 1):
        label = " | ".join(f"{k}={v}" for k, v in combo.items())
        print(f"[{i:3d}/{total}] {label} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            row = _run_combo(config, dataset, df, target_col, combo, n_permutations, seed)
            row["pass_5pp"] = row["permuted_mean_pp"] < pass_threshold
            dt = time.perf_counter() - t0
            mark = "✓" if row["pass_5pp"] else "✗"
            print(f"{mark}  permuted_mean={row['permuted_mean_pp']:.2f} pp | real={row['real_advantage_pct']:.2f}%  ({dt:.1f}s)")
        except Exception as exc:
            dt = time.perf_counter() - t0
            print(f"ERROR ({dt:.1f}s): {exc}")
            row = {**combo, "error": str(exc), "permuted_mean_pp": float("nan"), "pass_5pp": False}
        results.append(row)

    return results


def _save_csv(results: list[dict], path: Path) -> None:
    if not results:
        return
    # Collect all keys except the raw advantages list
    fieldnames = [k for k in results[0].keys() if k != "permuted_advantages"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def _fmt(val: Any, digits: int = 3) -> str:
    try:
        return f"{float(val):.{digits}f}"
    except Exception:
        return str(val)


def generate_html_report(
    dataset: str,
    results: list[dict],
    pass_threshold: float,
    n_permutations: int,
    grid: dict,
    output_path: Path,
) -> None:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    sorted_results = sorted(results, key=lambda r: r.get("permuted_mean_pp", float("inf")))
    passing = [r for r in results if r.get("pass_5pp") is True]

    param_keys = list(grid.keys())

    def row_class(r: dict) -> str:
        if r.get("pass_5pp"):
            return " class='pass-row'"
        return ""

    header_cells = "".join(
        f"<th>{k}</th>" for k in param_keys
    )
    header_extra = (
        "<th>Permuted mean (pp)</th>"
        "<th>Permuted std (pp)</th>"
        "<th>Real advantage (%)</th>"
        "<th>Baseline RMSE</th>"
        "<th>SCE RMSE</th>"
        "<th>Pass (&lt;5 pp)</th>"
    )

    body_rows = []
    for r in sorted_results:
        cells = "".join(f"<td>{r.get(k, '')}</td>" for k in param_keys)
        pm = r.get("permuted_mean_pp", float("nan"))
        ps = r.get("permuted_std_pp", float("nan"))
        ra = r.get("real_advantage_pct", float("nan"))
        br = r.get("baseline_rmse", "")
        sr = r.get("sce_rmse", "")
        ok = r.get("pass_5pp", False)
        badge = (
            "<span class='badge ok'>PASS</span>" if ok
            else "<span class='badge bad'>FAIL</span>"
        )
        body_rows.append(
            f"<tr{row_class(r)}>{cells}"
            f"<td class='num'>{_fmt(pm, 2)}</td>"
            f"<td class='num'>{_fmt(ps, 2)}</td>"
            f"<td class='num'>{_fmt(ra, 2)}</td>"
            f"<td class='num'>{br}</td>"
            f"<td class='num'>{sr}</td>"
            f"<td>{badge}</td>"
            "</tr>"
        )

    # Per-parameter summary: which values tend to reduce permuted mean
    param_summary_rows = []
    for key in param_keys:
        values = sorted(set(r.get(key) for r in results if key in r and "error" not in r))
        for val in values:
            subset = [r for r in results if r.get(key) == val and "error" not in r]
            if not subset:
                continue
            means = [r["permuted_mean_pp"] for r in subset]
            avg = float(np.mean(means))
            n_pass = sum(1 for r in subset if r.get("pass_5pp"))
            param_summary_rows.append(
                f"<tr><td><code>{key}</code></td><td>{val}</td>"
                f"<td class='num'>{avg:.2f}</td>"
                f"<td class='num'>{n_pass}/{len(subset)}</td></tr>"
            )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SCE Permuted-Target Sweep: {dataset}</title>
  <style>
    :root {{
      --bg: #f6f4ef;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #52606d;
      --line: #d9e2ec;
      --ok-bg: #e6f4ea; --ok-fg: #137333;
      --bad-bg: #fce8e6; --bad-fg: #c5221f;
      --pass-row: #f0faf2;
      --accent: #0b7285;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; background: var(--bg); color: var(--ink); font-size: 14px; }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 28px 20px 48px; }}
    h1 {{ margin: 0 0 6px; font-size: 26px; }}
    .sub {{ color: var(--muted); margin-bottom: 24px; font-size: 13px; }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); margin-bottom: 24px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }}
    .card h2 {{ margin: 0 0 6px; font-size: 15px; color: var(--accent); }}
    .kpi {{ font-size: 26px; font-weight: 700; margin: 2px 0; }}
    .hint {{ color: var(--muted); font-size: 12px; }}
    section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 16px; margin-bottom: 18px; overflow-x: auto; }}
    section h2 {{ margin: 0 0 10px; font-size: 16px; color: var(--accent); }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; }}
    th {{ color: var(--muted); font-weight: 600; background: #fafbfc; position: sticky; top: 0; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .pass-row td {{ background: var(--pass-row); }}
    .badge {{ display: inline-block; border-radius: 999px; font-size: 11px; font-weight: 700; padding: 2px 9px; }}
    .ok {{ background: var(--ok-bg); color: var(--ok-fg); }}
    .bad {{ background: var(--bad-bg); color: var(--bad-fg); }}
    code {{ background: #f0f2f4; border-radius: 4px; padding: 1px 5px; font-size: 12px; }}
    .threshold-note {{ background: #fff8dc; border: 1px solid #e0c060; border-radius: 8px; padding: 10px 14px; margin-bottom: 20px; font-size: 13px; }}
  </style>
</head>
<body>
<div class="wrap">
  <h1>SCE Permuted-Target Grid-Search Sweep</h1>
  <div class="sub">Dataset: <strong>{dataset}</strong> &nbsp;|&nbsp; {len(results)} combinations &nbsp;|&nbsp;
    {n_permutations} permutations each &nbsp;|&nbsp; Generated: {generated_at}</div>

  <div class="threshold-note">
    <strong>Pass criterion:</strong> permuted mean advantage &lt; <strong>{pass_threshold} pp</strong>.
    This checks whether SCE produces real signal beyond random noise — a lower value under permutation is healthier.
    A high permuted advantage means the model exploits cross-sample patterns even on a shuffled target,
    which is a sign of overfitting or leakage in the enrichment.<br><br>
    <strong>Real advantage</strong> is how much RMSE improves on the <em>real</em> target — this should be high.
    You want: real_advantage high AND permuted_mean low.
  </div>

  <div class="grid">
    <div class="card">
      <h2>Combinations tested</h2>
      <div class="kpi">{len(results)}</div>
      <div class="hint">{' × '.join(f'{k}: {len(v)}' for k, v in grid.items())}</div>
    </div>
    <div class="card">
      <h2>Passing (&lt;{pass_threshold} pp)</h2>
      <div class="kpi" style="color:{'#137333' if passing else '#c5221f'}">{len(passing)}</div>
      <div class="hint">out of {len(results)} combinations</div>
    </div>
    <div class="card">
      <h2>Best permuted mean</h2>
      <div class="kpi">{_fmt(sorted_results[0].get('permuted_mean_pp', 'n/a'), 2)} pp</div>
      <div class="hint">lowest = healthiest</div>
    </div>
    <div class="card">
      <h2>Best real advantage</h2>
      <div class="kpi">{_fmt(max((r.get('real_advantage_pct', 0) for r in results), default=0), 2)}%</div>
      <div class="hint">highest = most useful</div>
    </div>
  </div>

  <section>
    <h2>Parameter Sensitivity Summary</h2>
    <p style="color:var(--muted);font-size:12px;margin:0 0 8px">
      For each parameter value, the average permuted mean pp across all combos that include that value.
      Lower = that value tends to reduce leakage signal.
    </p>
    <table>
      <thead><tr><th>Parameter</th><th>Value</th><th>Avg permuted mean (pp)</th><th>Passing combos</th></tr></thead>
      <tbody>{''.join(param_summary_rows)}</tbody>
    </table>
  </section>

  <section>
    <h2>All Combinations (sorted by permuted mean ↑)</h2>
    <table>
      <thead>
        <tr>{header_cells}{header_extra}</tr>
      </thead>
      <tbody>
        {''.join(body_rows)}
      </tbody>
    </table>
  </section>

  <section>
    <h2>How to iterate on these parameters</h2>
    <table>
      <thead><tr><th>Parameter</th><th>TOML section</th><th>Effect on permuted signal</th><th>Cost</th></tr></thead>
      <tbody>
        <tr>
          <td><code>min_group_size</code></td>
          <td><code>[sce]</code></td>
          <td>Larger = statistics from bigger, more stable groups = less noise exploitation. Try 5 → 8 → 12.</td>
          <td>May reduce features for rare categories. Mild RMSE trade-off.</td>
        </tr>
        <tr>
          <td><code>include_interactions</code></td>
          <td><code>[sce]</code></td>
          <td>Disabling removes cross-category interaction features, which often carry the most random signal.</td>
          <td>Can lose 10–30% of real advantage on datasets with strong category interactions.</td>
        </tr>
        <tr>
          <td><code>include_fold_variance</code></td>
          <td><code>[sce]</code></td>
          <td>Disabling removes _fold_std/_lower/_upper features. These sometimes memorize fold-specific noise.</td>
          <td>Small RMSE trade-off; especially useful when n_folds is low.</td>
        </tr>
        <tr>
          <td><code>missing_threshold</code></td>
          <td><code>[run.feature_pruning]</code></td>
          <td>Lower = drop more sparse features. Removes fragile context stats that only fire on a small fraction of rows.</td>
          <td>Reduces feature count, may lose some rare-category context.</td>
        </tr>
        <tr>
          <td><code>n_folds</code></td>
          <td><code>[sce]</code></td>
          <td>Fewer folds = smaller out-of-fold windows = less information per fold = noisier, but also less overfitting.</td>
          <td>Not in this sweep — add to grid if above params don't solve it.</td>
        </tr>
        <tr>
          <td><code>cross_fit_strategy</code></td>
          <td><code>[sce]</code></td>
          <td>Switching from <code>random</code> to <code>rolling</code> enforces temporal discipline in fold assignment.</td>
          <td>Only relevant for time-series datasets. No effect on random splits.</td>
        </tr>
      </tbody>
    </table>
  </section>
</div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid-search sweep over SCE params (permuted-target signal)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--n-permutations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pass-threshold", type=float, default=5.0)
    args = parser.parse_args()

    results = run_sweep(
        dataset=args.dataset,
        grid=GRID,
        n_permutations=args.n_permutations,
        seed=args.seed,
        pass_threshold=args.pass_threshold,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"{args.dataset}_perm_sweep_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save raw JSON
    raw_path = out_dir / "sweep_results.json"
    raw_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # Save CSV (without the raw advantage list)
    csv_path = out_dir / "sweep_results.csv"
    _save_csv(results, csv_path)

    # Generate HTML report
    html_path = out_dir / "sweep_report.html"
    generate_html_report(
        dataset=args.dataset,
        results=results,
        pass_threshold=args.pass_threshold,
        n_permutations=args.n_permutations,
        grid=GRID,
        output_path=html_path,
    )

    print(f"\n{'='*60}")
    print(f"Results saved to: {out_dir}")
    print(f"  CSV:    {csv_path.name}")
    print(f"  JSON:   {raw_path.name}")
    print(f"  Report: {html_path.name}")
    print(f"{'='*60}")
    print(f"\nOpen report: {html_path}")
    print(str(html_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
