"""
@module: tests.equity.test_forward_target_isolation
@depends: equity.diagnostics.forward_target_isolation
@exports:
@data_flow: clean/leaky feature frames -> run_forward_target_isolation -> pass flag

S5.3 forward-target isolation guard tests. Clean frame PASSes; injected
``ret_h5`` FAIL + CLI exit 1.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from equity.diagnostics.forward_target_isolation import (
    _resolve_under_project_root,
    main,
    run_forward_target_isolation,
)


def _clean_features(n: int = 20) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [f"TK{i % 3}" for i in range(n)],
            "period_close_ts": pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC"),
            "sma_5": range(n),
            "ret_1d_log": [0.01] * n,  # PAST feature — allowed
            "sector_ret_1d_mean": [0.0] * n,
            "volume": range(n),
        }
    )


def test_clean_frame_passes() -> None:
    result = run_forward_target_isolation(_clean_features())
    assert result["pass"] is True
    assert result["n_violations"] == 0
    assert result["forbidden_found"] == []


def test_injected_ret_h5_fails() -> None:
    dirty = _clean_features()
    dirty["ret_h5"] = 0.02
    result = run_forward_target_isolation(dirty, horizons=(1, 5, 10, 21, 63))
    assert result["pass"] is False
    assert result["n_violations"] >= 1
    assert "ret_h5" in result["forbidden_found"]
    types = {v["type"] for v in result["violations"]}
    assert "forward_target_in_features" in types


def test_pred_hn_and_quantile_also_flagged() -> None:
    dirty = _clean_features()
    dirty["pred_h1"] = 0.0
    dirty["pred_h10_q05"] = -0.1
    dirty["resid_h21"] = 0.0
    result = run_forward_target_isolation(dirty)
    assert result["pass"] is False
    found = set(result["forbidden_found"])
    assert "pred_h1" in found
    assert "pred_h10_q05" in found
    assert "resid_h21" in found


def test_pred_sector_allowed_flag() -> None:
    frame = _clean_features()
    frame["pred_sector_h5"] = 0.01
    blocked = run_forward_target_isolation(frame, allowed_pred_sector=False)
    assert blocked["pass"] is False
    allowed = run_forward_target_isolation(frame, allowed_pred_sector=True)
    assert allowed["pass"] is True


def test_cli_exits_nonzero_on_leak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dirty = _clean_features()
    dirty["ret_h5"] = 0.02
    feat_path = tmp_path / "leaky.parquet"
    dirty.to_parquet(feat_path, index=False)

    # Point PROJECT_ROOT containment at tmp so --output can write there.
    import equity.diagnostics.forward_target_isolation as mod

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path / "results")

    # features path is absolute under tmp_path; _resolve only wraps --output.
    # main reads --features with pd.read_parquet directly (absolute ok).
    out = tmp_path / "report.json"
    code = main(
        [
            "--features",
            str(feat_path),
            "--output",
            str(out.name),  # relative under PROJECT_ROOT=tmp_path
        ]
    )
    assert code == 1
    report_path = tmp_path / out.name
    assert report_path.is_file()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["pass"] is False


def test_cli_exits_zero_on_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clean = _clean_features()
    feat_path = tmp_path / "clean.parquet"
    clean.to_parquet(feat_path, index=False)

    import equity.diagnostics.forward_target_isolation as mod

    monkeypatch.setattr(mod, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mod, "RESULTS_DIR", tmp_path / "results")

    code = main(["--features", str(feat_path), "--output", "ok.json"])
    assert code == 0


def test_resolve_refuses_traversal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="path-traversal"):
        _resolve_under_project_root("../etc/passwd", project_root=tmp_path)
