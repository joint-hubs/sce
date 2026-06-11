"""Generate a compact end-to-end HTML report from latest diagnostics and search artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _latest_file(path_glob: str) -> Path | None:
    files = sorted(RESULTS_DIR.glob(path_glob), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _latest_dir(path_glob: str) -> Path | None:
    dirs = [p for p in RESULTS_DIR.glob(path_glob) if p.is_dir()]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0] if dirs else None


def _fmt_float(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _status_badge(ok: bool) -> str:
    if ok:
        return "<span class='badge ok'>PASS</span>"
    return "<span class='badge bad'>FAIL</span>"


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def generate_report(dataset: str, output: Path | None = None) -> Path:
    perm_path = _latest_file(f"diagnostics/{dataset}/permuted_target_*.json")
    shuf_all_path = _latest_file(f"diagnostics/{dataset}/shuffled_groups_all_*.json")
    shuf_col_path = _latest_file(f"diagnostics/{dataset}/shuffled_groups_per-column_*.json")
    crossfit_path = _latest_file(f"diagnostics/{dataset}/crossfit_ab_*.json")
    search_dir = _latest_dir(f"{dataset}_search_*")

    if not all([perm_path, shuf_all_path, shuf_col_path, crossfit_path, search_dir]):
        missing = []
        if not perm_path:
            missing.append("permuted_target")
        if not shuf_all_path:
            missing.append("shuffled_groups_all")
        if not shuf_col_path:
            missing.append("shuffled_groups_per-column")
        if not crossfit_path:
            missing.append("crossfit_ab")
        if not search_dir:
            missing.append("search_dir")
        raise FileNotFoundError(f"Missing required artifacts for {dataset}: {', '.join(missing)}")

    metadata_path = search_dir / "data" / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.json in {search_dir}")

    perm = _load_json(perm_path)
    shuf_all = _load_json(shuf_all_path)
    shuf_col = _load_json(shuf_col_path)
    crossfit = _load_json(crossfit_path)
    meta = _load_json(metadata_path)

    search_metrics = meta.get("metrics", {})
    search_diag = meta.get("diagnostics", {}).get("feature_dominance", {})

    diagnostics_ok = {
        "Permuted target": bool(perm.get("pass", False)),
        "Shuffled groups (all)": bool(shuf_all.get("pass", False)),
        "Shuffled groups (per-column)": bool(shuf_col.get("pass", False)),
        "Cross-fit A/B": float(crossfit.get("leakage_signal_pp", 0.0)) >= 0.0,
    }

    rows = []
    for name, ok in diagnostics_ok.items():
        rows.append(f"<tr><td>{name}</td><td>{_status_badge(ok)}</td></tr>")

    per_col_rows = []
    for col, payload in (shuf_col.get("per_column", {}) or {}).items():
        per_col_rows.append(
            "<tr>"
            f"<td>{col}</td>"
            f"<td>{_fmt_float(payload.get('mean_advantage'), 3)}%</td>"
            f"<td>{len(payload.get('advantages', []))}</td>"
            "</tr>"
        )
    if not per_col_rows:
        per_col_rows.append("<tr><td colspan='3'>No per-column details available</td></tr>")

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>SCE End-to-End Report: {dataset}</title>
  <style>
    :root {{
      --bg: #f4f2ed;
      --panel: #ffffff;
      --ink: #1f2933;
      --muted: #52606d;
      --line: #d9e2ec;
      --ok-bg: #e6f4ea;
      --ok-fg: #137333;
      --bad-bg: #fce8e6;
      --bad-fg: #c5221f;
      --accent: #0b7285;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "Segoe UI", Tahoma, sans-serif; background: linear-gradient(135deg, #f8f6f1 0%, #ecf7fa 100%); color: var(--ink); }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 28px 20px 36px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    .sub {{ color: var(--muted); margin-bottom: 18px; }}
    .grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05); }}
    .card h2 {{ margin: 0 0 8px; font-size: 17px; color: var(--accent); }}
    .kpi {{ font-size: 28px; font-weight: 700; margin: 4px 0 2px; }}
    .hint {{ color: var(--muted); font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 8px; text-align: left; font-size: 14px; }}
    th {{ color: var(--muted); font-weight: 600; }}
    .badge {{ display: inline-block; border-radius: 999px; font-size: 12px; font-weight: 700; padding: 3px 10px; }}
    .ok {{ background: var(--ok-bg); color: var(--ok-fg); }}
    .bad {{ background: var(--bad-bg); color: var(--bad-fg); }}
    .artifacts a {{ color: #0a58ca; text-decoration: none; }}
    .artifacts a:hover {{ text-decoration: underline; }}
    .block {{ margin-top: 14px; }}
    @media (max-width: 700px) {{ .wrap {{ padding: 18px 12px 26px; }} h1 {{ font-size: 24px; }} }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>SCE End-to-End Quick Report</h1>
    <div class=\"sub\">Dataset: <strong>{dataset}</strong> | Generated: {generated_at} | Mode: quick smoke run</div>

    <div class=\"grid\">
      <section class=\"card\">
        <h2>Search Outcome</h2>
        <div class=\"kpi\">{_fmt_float(search_metrics.get('rmse_improvement_pct'))}%</div>
        <div class=\"hint\">RMSE improvement vs baseline</div>
        <div class=\"block\">Baseline RMSE: <strong>{_fmt_float(search_metrics.get('baseline_rmse'))}</strong></div>
        <div>SCE RMSE: <strong>{_fmt_float(search_metrics.get('sce_rmse'))}</strong></div>
      </section>

      <section class=\"card\">
        <h2>Permuted Target</h2>
        <div class=\"kpi\">{_fmt_float(perm.get('sce_advantage_permuted_mean'))}%</div>
        <div class=\"hint\">Expected: low advantage after target shuffle</div>
        <div class=\"block\">Status: {_status_badge(bool(perm.get('pass', False)))}</div>
      </section>

      <section class=\"card\">
        <h2>Cross-fit A/B</h2>
        <div class=\"kpi\">{_fmt_float(crossfit.get('leakage_signal_pp'))} pp</div>
        <div class=\"hint\">Leakage signal pp (>= 0 preferred)</div>
        <div class=\"block\">Status: {_status_badge(float(crossfit.get('leakage_signal_pp', 0.0)) >= 0.0)}</div>
      </section>

      <section class=\"card\">
        <h2>Feature Dominance</h2>
        <div class=\"kpi\">{_fmt_float(search_diag.get('top_k_share_pct', 0.0))}%</div>
        <div class=\"hint\">Top-3 importance share</div>
        <div class=\"block\">Status: {_status_badge(not bool(search_diag.get('dominated', False)))}</div>
      </section>
    </div>

    <section class=\"card block\">
      <h2>Diagnostic Gates</h2>
      <table>
        <thead><tr><th>Diagnostic</th><th>Status</th></tr></thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </section>

    <section class=\"card block\">
      <h2>Shuffled Groups: Per-Column Breakdown</h2>
      <table>
        <thead><tr><th>Column</th><th>Mean advantage after shuffle</th><th>Permutations</th></tr></thead>
        <tbody>
          {''.join(per_col_rows)}
        </tbody>
      </table>
    </section>

    <section class=\"card block artifacts\">
      <h2>Artifacts</h2>
      <table>
        <thead><tr><th>Type</th><th>Path</th></tr></thead>
        <tbody>
          <tr><td>Search report dir</td><td><a href=\"{_rel(search_dir)}\">{_rel(search_dir)}</a></td></tr>
          <tr><td>Search metadata</td><td><a href=\"{_rel(metadata_path)}\">{_rel(metadata_path)}</a></td></tr>
          <tr><td>Permuted target</td><td><a href=\"{_rel(perm_path)}\">{_rel(perm_path)}</a></td></tr>
          <tr><td>Shuffled groups (all)</td><td><a href=\"{_rel(shuf_all_path)}\">{_rel(shuf_all_path)}</a></td></tr>
          <tr><td>Shuffled groups (per-column)</td><td><a href=\"{_rel(shuf_col_path)}\">{_rel(shuf_col_path)}</a></td></tr>
          <tr><td>Cross-fit A/B</td><td><a href=\"{_rel(crossfit_path)}\">{_rel(crossfit_path)}</a></td></tr>
        </tbody>
      </table>
    </section>
  </div>
</body>
</html>
"""

    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = RESULTS_DIR / f"{dataset}_e2e_report_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "report.html"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(html, encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate end-to-end HTML report from latest artifacts")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None, help="Optional output HTML path")
    args = parser.parse_args()

    out_path = generate_report(args.dataset, Path(args.output) if args.output else None)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
