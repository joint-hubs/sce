"""
@module: scripts.night_report
@depends: results/
@exports: full overnight report (markdown)
@data_flow: results artifacts -> consolidated markdown report

Collects everything produced by an overnight batch (single runs, searches,
categorical-mode comparisons, diagnostics) into one markdown report so the
numbers can be reviewed and pushed in a single pass.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
DIAG_DIR = RESULTS_DIR / "diagnostics"

ACTIVE_DATASETS = [
    "rental_poland_short",
    "melbourne_housing",
    "m5_store_dept_daily",
    "walmart_weekly",
    "rossmann_daily",
]
EXPERIMENTAL_DATASETS = [
    "experimental/sales_uae_transactions",
    "experimental/rental_uae_contracts",
]


def _latest_dir(pattern: str) -> Path | None:
    dirs = sorted(RESULTS_DIR.glob(pattern))
    return dirs[-1] if dirs else None


def _load_json(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def section_experiments() -> list[str]:
    lines = ["## 1. Single experiments (report-grade)", ""]
    data = _load_json(RESULTS_DIR / "experiment_results.json")
    if not data:
        return lines + ["_experiment_results.json missing._", ""]

    lines += [
        "| Dataset | Baseline RMSE | SCE RMSE | RMSE Δ% | R² Δpp | Promoted |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in data:
        name = r.get("dataset", "?")
        promoted = "?"
        latest = _latest_dir(f"{name}_single_*")
        if latest:
            meta = _load_json(latest / "data" / "metadata.json")
            if meta:
                promoted = str(meta.get("promotion", {}).get("promoted_to_report_grade"))
        lines.append(
            f"| {name} | {r.get('baseline_rmse', 0):,.2f} | {r.get('sce_rmse', 0):,.2f} "
            f"| {r.get('rmse_improvement_pct', 0):+.2f}% | {r.get('r2_improvement_pct', 0):+.2f} "
            f"| {promoted} |"
        )
    return lines + [""]


def section_search() -> list[str]:
    lines = ["## 2. Feature-combination search (validation-selected, test touched once)", ""]
    any_found = False
    for name in ACTIVE_DATASETS:
        latest = _latest_dir(f"{name}_search_*")
        if latest is None:
            lines.append(f"- **{name}**: _no search results_")
            continue
        any_found = True
        final = latest / "data" / "final_test_results.csv"
        if final.exists():
            df = pd.read_csv(final)
            best = df.sort_values("rmse").iloc[0] if "rmse" in df.columns else df.iloc[0]
            rmse = best.get("rmse", float("nan"))
            val_rmse = best.get("val_rmse", float("nan"))
            lines.append(
                f"- **{name}** ({latest.name}): best test RMSE = {rmse:,.2f} "
                f"(val RMSE = {val_rmse:,.2f}), {len(df)} winner(s) evaluated on test"
            )
        else:
            lines.append(f"- **{name}** ({latest.name}): _final_test_results.csv missing_")
    if not any_found:
        lines.append("")
        lines.append("_No fresh search runs found._")
    return lines + [""]


def section_categorical_compare() -> list[str]:
    lines = ["## 3. Categorical-mode comparison (manual vs auto)", ""]
    for name in ACTIVE_DATASETS:
        latest = _latest_dir(f"{name}_categorical_compare_*")
        if latest is None:
            lines.append(f"- **{name}**: _no comparison run_")
            continue
        csv_path = latest / "data" / "categorical_mode_comparison.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            lines.append(f"- **{name}** ({latest.name}): {len(df)} rows")
            for _, row in df.iterrows():
                mode = row.get("categorical_mode", row.get("mode", "?"))
                imp = row.get("rmse_improvement_pct", float("nan"))
                lines.append(f"    - {mode}: RMSE Δ {imp:+.2f}%")
        else:
            lines.append(f"- **{name}** ({latest.name}): _comparison csv missing_")
    return lines + [""]


def section_diagnostics() -> list[str]:
    lines = ["## 4. Leakage diagnostics (latest per dataset)", ""]
    lines += [
        "| Dataset | Diagnostic | Rows | Key metric | Pass |",
        "|---|---|---:|---|---|",
    ]
    all_names = ACTIVE_DATASETS + EXPERIMENTAL_DATASETS
    for name in all_names:
        diag_dir = DIAG_DIR / name
        if not diag_dir.exists():
            lines.append(f"| {name} | — | — | _no diagnostics dir_ | — |")
            continue
        for diag, pattern in [
            ("permuted_target", "permuted_target_*.json"),
            ("shuffled_groups", "shuffled_groups_*.json"),
            ("crossfit_ab", "crossfit_ab_*.json"),
        ]:
            files = sorted(diag_dir.glob(pattern))
            if not files:
                lines.append(f"| {name} | {diag} | — | _missing_ | — |")
                continue
            payload = _load_json(files[-1]) or {}
            rows = payload.get("evaluated_rows", "?")
            if diag == "permuted_target":
                key = f"perm adv mean = {payload.get('sce_advantage_permuted_mean', float('nan')):+.2f}%"
            elif diag == "shuffled_groups":
                key = f"shuf adv mean = {payload.get('sce_advantage_shuffled_mean', float('nan')):+.2f}%"
            else:
                key = f"leakage signal = {payload.get('leakage_signal_pp', float('nan')):+.2f}pp"
            passed = payload.get("pass", "n/a")
            lines.append(
                f"| {name} | {diag} | {rows:,} | {key} | {passed} |"
                if isinstance(rows, int)
                else f"| {name} | {diag} | {rows} | {key} | {passed} |"
            )
    return lines + [""]


def section_figures() -> list[str]:
    lines = ["## 5. Figure artifacts (docs/figures/paper)", ""]
    fig_dir = ROOT / "docs" / "figures" / "paper"
    if not fig_dir.exists():
        return lines + ["_figures dir missing_", ""]
    cutoff = datetime.now().timestamp() - 24 * 3600
    fresh, stale = [], []
    for f in sorted(fig_dir.glob("*")):
        (fresh if f.stat().st_mtime >= cutoff else stale).append(f.name)
    lines.append(f"- regenerated in the last 24h ({len(fresh)}): " + (", ".join(fresh) or "—"))
    lines.append(f"- older ({len(stale)}): " + (", ".join(stale) or "—"))
    return lines + [""]


def main() -> int:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    lines = [
        f"# Overnight full report — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "_Auto-generated by scripts/night_report.py. Every number below comes from",
        "artifacts in results/; promotion and pass fields are read from run metadata._",
        "",
    ]
    lines += section_experiments()
    lines += section_search()
    lines += section_categorical_compare()
    lines += section_diagnostics()
    lines += section_figures()

    out = RESULTS_DIR / f"night_report_{ts}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
