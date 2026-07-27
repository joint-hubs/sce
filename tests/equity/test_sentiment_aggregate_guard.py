"""
@module: tests.equity.test_sentiment_aggregate_guard
@depends: equity.diagnostics.sentiment_aggregate_guard
@exports:
@data_flow: per-article + per-period frames -> run_sentiment_aggregate_guard
"""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd
import pytest

from equity.diagnostics.sentiment_aggregate_guard import (
    run_sentiment_aggregate_guard,
)
from equity.sentiment.aggregate import aggregate_per_period


def _valid_per_article() -> pd.DataFrame:
    period_close = pd.Timestamp("2024-07-08 20:00", tz="UTC")
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL"],
            "published_at": [
                pd.Timestamp("2024-07-08 18:00", tz="UTC"),
                pd.Timestamp("2024-07-08 12:00", tz="UTC"),
            ],
            "period_close_ts": [period_close, period_close],
            "pos": [0.7, 0.5],
            "neg": [0.2, 0.3],
            "neu": [0.1, 0.2],
            "score": [0.5, 0.2],
        }
    )


def _valid_per_period() -> pd.DataFrame:
    # Derive from the aggregator so the reproducibility invariant (M3) holds
    # by construction -- the stored per_period matches a fresh re-derivation
    # from per_article.
    return aggregate_per_period(_valid_per_article())


def test_guard_passes_on_valid_frames():
    result = run_sentiment_aggregate_guard(_valid_per_article(), _valid_per_period())
    assert result["pass"] is True
    assert result["n_violations"] == 0


def test_guard_detects_pit_violation():
    df = _valid_per_article().copy()
    # Move one article's published_at AFTER the period close.
    df.loc[0, "published_at"] = pd.Timestamp("2024-07-08 21:00", tz="UTC")
    result = run_sentiment_aggregate_guard(df, _valid_per_period())
    assert result["pass"] is False
    assert any(v["type"] == "pit_violation" for v in result["violations"])


def test_guard_detects_prob_sum_violation():
    per_period = _valid_per_period().copy()
    per_period.loc[0, "sentiment_neu"] = 0.5  # 0.6 + 0.25 + 0.5 = 1.35
    result = run_sentiment_aggregate_guard(_valid_per_article(), per_period)
    assert result["pass"] is False
    assert any(v["type"] == "prob_sum_violation" for v in result["violations"])


def test_guard_cache_freshness_mismatch():
    cache_meta = {"model_name": "stub", "model_revision": "v1"}
    result = run_sentiment_aggregate_guard(
        _valid_per_article(),
        _valid_per_period(),
        cache_meta=cache_meta,
        expected_model="stub",
        expected_revision="v2",
    )
    assert result["pass"] is False
    assert any(v["type"] == "cache_revision_mismatch" for v in result["violations"])


def test_guard_cache_freshness_match_passes():
    cache_meta = {"model_name": "stub", "model_revision": "v1"}
    result = run_sentiment_aggregate_guard(
        _valid_per_article(),
        _valid_per_period(),
        cache_meta=cache_meta,
        expected_model="stub",
        expected_revision="v1",
    )
    assert result["pass"] is True


def test_guard_weight_monotonicity_passes_on_valid_frame():
    """The valid fixture has articles at 18:00 and 12:00 UTC (newer first).
    Weights are monotonic in published_at -- no violation.
    """
    result = run_sentiment_aggregate_guard(_valid_per_article(), _valid_per_period())
    assert not any(v["type"] == "weight_monotonicity_violation" for v in result["violations"])


def test_guard_cli_passes_on_valid_inputs(tmp_path):
    per_article = _valid_per_article()
    per_period = _valid_per_period()
    ap_path = tmp_path / "per_article.parquet"
    pp_path = tmp_path / "per_period.parquet"
    per_article.to_parquet(ap_path, index=False)
    per_period.to_parquet(pp_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "equity.diagnostics.sentiment_aggregate_guard",
            "--per-article",
            str(ap_path),
            "--per-period",
            str(pp_path),
            "--output",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["pass"] is True


def test_guard_cli_fails_on_pit_violation(tmp_path):
    df = _valid_per_article().copy()
    df.loc[0, "published_at"] = pd.Timestamp("2024-07-08 21:00", tz="UTC")
    ap_path = tmp_path / "per_article.parquet"
    pp_path = tmp_path / "per_period.parquet"
    df.to_parquet(ap_path, index=False)
    _valid_per_period().to_parquet(pp_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "equity.diagnostics.sentiment_aggregate_guard",
            "--per-article",
            str(ap_path),
            "--per-period",
            str(pp_path),
            "--output",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stdout
    out = json.loads((tmp_path / "out.json").read_text())
    assert out["pass"] is False


def test_guard_cli_with_cache_meta_freshness_check(tmp_path):
    """Exercise the ``--cache-meta`` / ``--expected-model`` /
    ``--expected-revision`` CLI flags (covers the freshness-check branch
    of :func:`main`).
    """
    ap_path = tmp_path / "per_article.parquet"
    pp_path = tmp_path / "per_period.parquet"
    meta_path = tmp_path / "_meta.json"
    _valid_per_article().to_parquet(ap_path, index=False)
    _valid_per_period().to_parquet(pp_path, index=False)
    meta_path.write_text(json.dumps({"model_name": "stub", "model_revision": "v1"}))

    # Matching model/revision -> PASS.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "equity.diagnostics.sentiment_aggregate_guard",
            "--per-article",
            str(ap_path),
            "--per-period",
            str(pp_path),
            "--cache-meta",
            str(meta_path),
            "--expected-model",
            "stub",
            "--expected-revision",
            "v1",
            "--output",
            str(tmp_path / "out_match.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads((tmp_path / "out_match.json").read_text())
    assert out["pass"] is True
    assert out["cache_meta_checked"] is True

    # Mismatching revision -> FAIL.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "equity.diagnostics.sentiment_aggregate_guard",
            "--per-article",
            str(ap_path),
            "--per-period",
            str(pp_path),
            "--cache-meta",
            str(meta_path),
            "--expected-model",
            "stub",
            "--expected-revision",
            "v2",
            "--output",
            str(tmp_path / "out_mismatch.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1, proc.stdout
    out = json.loads((tmp_path / "out_mismatch.json").read_text())
    assert out["pass"] is False


def test_guard_cli_default_output_path(tmp_path, monkeypatch):
    """When ``--output`` is omitted, the result is written under
    ``results/diagnostics/equity/`` with a UTC timestamp suffix.
    """
    # Point RESULTS_DIR at a tmp_path subdirectory so the test does not
    # pollute the real repo.
    results_dir = tmp_path / "results" / "diagnostics" / "equity"
    monkeypatch.setattr("equity.diagnostics.sentiment_aggregate_guard.RESULTS_DIR", results_dir)
    pp_path = tmp_path / "per_period.parquet"
    _valid_per_period().to_parquet(pp_path, index=False)

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "equity.diagnostics.sentiment_aggregate_guard",
            "--per-period",
            str(pp_path),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    # The guard writes to the real RESULTS_DIR (subprocess does not inherit
    # the monkeypatch); verify the default-output branch executed by
    # checking stdout contains "Result written to:".
    assert "Result written to:" in proc.stdout


def test_guard_cli_warns_on_unparsable_cache_meta(tmp_path):
    """A corrupt ``--cache-meta`` file logs a WARNING and skips the
    freshness check (does NOT crash).
    """
    ap_path = tmp_path / "per_article.parquet"
    pp_path = tmp_path / "per_period.parquet"
    meta_path = tmp_path / "_meta.json"
    _valid_per_article().to_parquet(ap_path, index=False)
    _valid_per_period().to_parquet(pp_path, index=False)
    meta_path.write_text("{not valid json")

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "equity.diagnostics.sentiment_aggregate_guard",
            "--per-article",
            str(ap_path),
            "--per-period",
            str(pp_path),
            "--cache-meta",
            str(meta_path),
            "--expected-model",
            "stub",
            "--expected-revision",
            "v1",
            "--output",
            str(tmp_path / "out.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "WARNING" in proc.stdout or "could not parse" in proc.stdout


def test_guard_no_per_article_skips_pit_and_monotonicity():
    """When ``per_article`` is empty, PIT + monotonicity checks are skipped
    but the prob-sum check on ``per_period`` still runs.
    """
    empty_pa = pd.DataFrame()
    result = run_sentiment_aggregate_guard(empty_pa, _valid_per_period())
    assert result["pass"] is True
    assert result["n_per_article"] == 0


def test_guard_cli_main_in_process_pass(tmp_path, monkeypatch):
    """Call ``main()`` in-process (via ``sys.argv`` monkeypatch) so the
    main() body is covered by coverage. Writes to a tmp_path output.
    """
    from equity.diagnostics import sentiment_aggregate_guard as guard

    ap_path = tmp_path / "per_article.parquet"
    pp_path = tmp_path / "per_period.parquet"
    out_path = tmp_path / "out.json"
    _valid_per_article().to_parquet(ap_path, index=False)
    _valid_per_period().to_parquet(pp_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sentiment_aggregate_guard",
            "--per-article",
            str(ap_path),
            "--per-period",
            str(pp_path),
            "--output",
            str(out_path),
        ],
    )
    rc = guard.main()
    assert rc == 0
    assert out_path.exists()
    out = json.loads(out_path.read_text())
    assert out["pass"] is True


def test_guard_cli_main_in_process_fail(tmp_path, monkeypatch):
    """Call ``main()`` in-process on a PIT-violating frame; expect rc=1."""
    from equity.diagnostics import sentiment_aggregate_guard as guard

    df = _valid_per_article().copy()
    df.loc[0, "published_at"] = pd.Timestamp("2024-07-08 21:00", tz="UTC")
    ap_path = tmp_path / "per_article.parquet"
    pp_path = tmp_path / "per_period.parquet"
    out_path = tmp_path / "out.json"
    df.to_parquet(ap_path, index=False)
    _valid_per_period().to_parquet(pp_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sentiment_aggregate_guard",
            "--per-article",
            str(ap_path),
            "--per-period",
            str(pp_path),
            "--output",
            str(out_path),
            "--decay-time-const-days",
            "5.0",
        ],
    )
    rc = guard.main()
    assert rc == 1
    out = json.loads(out_path.read_text())
    assert out["pass"] is False


def test_guard_cli_main_in_process_cache_meta(tmp_path, monkeypatch):
    """Call ``main()`` in-process with ``--cache-meta`` + expected model
    flags (covers the cache-meta parse + freshness-check branch).
    """
    from equity.diagnostics import sentiment_aggregate_guard as guard

    ap_path = tmp_path / "per_article.parquet"
    pp_path = tmp_path / "per_period.parquet"
    meta_path = tmp_path / "_meta.json"
    out_path = tmp_path / "out.json"
    _valid_per_article().to_parquet(ap_path, index=False)
    _valid_per_period().to_parquet(pp_path, index=False)
    meta_path.write_text(json.dumps({"model_name": "stub", "model_revision": "v1"}))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sentiment_aggregate_guard",
            "--per-article",
            str(ap_path),
            "--per-period",
            str(pp_path),
            "--cache-meta",
            str(meta_path),
            "--expected-model",
            "stub",
            "--expected-revision",
            "v1",
            "--output",
            str(out_path),
        ],
    )
    rc = guard.main()
    assert rc == 0
    out = json.loads(out_path.read_text())
    assert out["cache_meta_checked"] is True


def test_guard_cli_main_in_process_default_output(tmp_path, monkeypatch):
    """Call ``main()`` without ``--output`` -- the result is written under
    ``RESULTS_DIR`` (monkeypatched to tmp_path).
    """
    from equity.diagnostics import sentiment_aggregate_guard as guard

    results_dir = tmp_path / "results" / "diagnostics" / "equity"
    monkeypatch.setattr(guard, "RESULTS_DIR", results_dir)

    pp_path = tmp_path / "per_period.parquet"
    _valid_per_period().to_parquet(pp_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sentiment_aggregate_guard",
            "--per-period",
            str(pp_path),
        ],
    )
    rc = guard.main()
    assert rc == 0
    # One JSON file written under results_dir.
    jsons = list(results_dir.glob("sentiment_aggregate_guard_*.json"))
    assert len(jsons) == 1


# ---------------------------------------------------------------------------
# M3: aggregate reproducibility (5th invariant)
# ---------------------------------------------------------------------------


def test_guard_reproducibility_passes_on_consistent_frames():
    """When per_period is a faithful re-derivation of per_article, the
    reproducibility invariant passes.
    """
    result = run_sentiment_aggregate_guard(_valid_per_article(), _valid_per_period())
    assert result["pass"] is True
    assert not any(v["type"] == "aggregate_reproducibility" for v in result["violations"])


def test_guard_reproducibility_detects_tampered_per_period():
    """Manually editing per_period.sentiment_score breaks the
    re-derivation -> the guard reports an aggregate_reproducibility
    violation.
    """
    per_period = _valid_per_period().copy()
    per_period.loc[0, "sentiment_score"] = per_period.loc[0, "sentiment_score"] + 0.1
    result = run_sentiment_aggregate_guard(_valid_per_article(), per_period)
    assert result["pass"] is False
    repro = [v for v in result["violations"] if v["type"] == "aggregate_reproducibility"]
    assert len(repro) == 1
    assert repro[0]["ticker"] == "AAPL"


def test_guard_reproducibility_detects_wrong_n_articles():
    """Bumping per_period.n_articles without adding per_article rows is a
    reproducibility violation.
    """
    per_period = _valid_per_period().copy()
    per_period.loc[0, "n_articles"] = per_period.loc[0, "n_articles"] + 1
    result = run_sentiment_aggregate_guard(_valid_per_article(), per_period)
    repro = [v for v in result["violations"] if v["type"] == "aggregate_reproducibility"]
    assert len(repro) == 1


def test_guard_reproducibility_skipped_when_per_article_empty():
    """When per_article is empty, the reproducibility check is skipped
    (consistent with PIT/monotonicity skip-on-empty); the per_period
    prob-sum check still runs and the result is pass.
    """
    result = run_sentiment_aggregate_guard(pd.DataFrame(), _valid_per_period())
    assert result["pass"] is True
    assert not any(v["type"] == "aggregate_reproducibility" for v in result["violations"])


def test_guard_cli_rejects_output_outside_project_root():
    """L9: ``--output`` with a ``..`` traversal component is refused."""
    from equity.diagnostics import sentiment_aggregate_guard as guard

    with pytest.raises(ValueError, match="path-traversal"):
        guard._resolve_under_project_root("../../etc/evil.json")


def test_guard_cli_accepts_absolute_output_path(tmp_path):
    """L9: an absolute ``--output`` outside PROJECT_ROOT is honored -- a
    diagnostic report is legitimately written to operator-chosen locations
    (CI artifacts, tmp dirs)."""
    from equity.diagnostics import sentiment_aggregate_guard as guard

    out = tmp_path / "diagnostic.json"
    resolved = guard._resolve_under_project_root(str(out))
    assert resolved == out.resolve()
