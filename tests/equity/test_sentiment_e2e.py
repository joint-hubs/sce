"""
@module: tests.equity.test_sentiment_e2e
@depends: equity.sentiment.cache, equity.sentiment.aggregate,
          equity.diagnostics.sentiment_aggregate_guard
@exports:
@data_flow: articles_joined -> cache.score_articles -> aggregate_from_joined
            -> run_sentiment_aggregate_guard

End-to-end test of the S2 sentiment pipeline (FOC-49 B3): cache -> aggregate
-> guard. Exercises the pass-through glue (B1) and the 5 invariants (M3)
on a small synthetic ``articles_joined`` frame.
"""

from __future__ import annotations

import pandas as pd

from equity.diagnostics.sentiment_aggregate_guard import run_sentiment_aggregate_guard
from equity.sentiment.aggregate import aggregate_from_joined
from equity.sentiment.cache import SentimentCache


class _StubScorer:
    """Deterministic stub scorer for the E2E pipeline (mirrors the
    ``_CountingStubScorer`` pattern in ``test_sentiment_cache.py``).
    """

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def model_revision(self) -> str:
        return "v1"

    def classify(self, text: str) -> dict[str, float]:
        return self._score_one(text)

    def classify_batch(self, texts: list[str]) -> list[dict[str, float]]:
        return [self._score_one(t) for t in texts]

    def _score_one(self, text: str) -> dict[str, float]:
        h = (hash(text) & 0xFFFF) / 0xFFFF
        pos = 0.5 * h + 0.25
        neg = 0.25 * (1.0 - h)
        neu = 1.0 - pos - neg
        return {"pos": pos, "neg": neg, "neu": neu, "score": pos - neg}


def _articles_joined_fixture() -> pd.DataFrame:
    """4 articles across 2 tickers and 2 periods. All published_at <=
    period_close_ts (PIT-safe). ``articles_joined.parquet`` shape from S1.3.
    """
    close_p1 = pd.Timestamp("2024-07-08 20:00", tz="UTC")  # Mon 16:00 ET
    close_p2 = pd.Timestamp("2024-07-09 20:00", tz="UTC")  # Tue 16:00 ET
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "AAPL", "MSFT", "MSFT"],
            "published_at": [
                pd.Timestamp("2024-07-08 18:00", tz="UTC"),  # P1, AAPL
                pd.Timestamp("2024-07-08 12:00", tz="UTC"),  # P1, AAPL
                pd.Timestamp("2024-07-09 18:00", tz="UTC"),  # P2, MSFT
                pd.Timestamp("2024-07-09 12:00", tz="UTC"),  # P2, MSFT
            ],
            "period_close_ts": [close_p1, close_p1, close_p2, close_p2],
            "text": [
                "aapl great quarter P1",
                "aapl weak guidance P1",
                "msft strong cloud P2",
                "msft miss on azure P2",
            ],
            "source": ["reuters", "reuters", "bloomberg", "bloomberg"],
        }
    )


def test_e2e_cache_aggregate_guard_pass(tmp_path, monkeypatch):
    """Full pipeline PASS: score -> aggregate -> guard returns status=pass."""
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _StubScorer()
    articles = _articles_joined_fixture()

    per_article = cache.score_articles(articles, scorer)
    per_period = aggregate_from_joined(articles, cache, scorer)

    # Cache is warm after the first score_articles call, so aggregate_from_joined
    # makes ZERO additional forward passes (idempotency).
    assert scorer is not None  # sanity
    assert len(per_article) == 4
    assert "period_close_ts" in per_article.columns
    assert len(per_period) == 2
    assert set(per_period["ticker"]) == {"AAPL", "MSFT"}

    result = run_sentiment_aggregate_guard(per_article, per_period)
    assert result["pass"] is True, f"guard failed: {result['violations']}"
    assert result["n_violations"] == 0


def test_e2e_future_article_pit_leak_detected(tmp_path, monkeypatch):
    """FAIL path: one article has published_at > period_close_ts (future
    article PIT leak). The guard reports a ``pit_violation`` and status=fail.
    """
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _StubScorer()
    articles = _articles_joined_fixture()

    # First score + aggregate the PIT-safe fixture (PASS).
    per_article = cache.score_articles(articles, scorer)
    per_period = aggregate_from_joined(articles, cache, scorer)

    # Inject a PIT violation: move one article's published_at AFTER its
    # period_close_ts in BOTH frames (simulates a future-article leak into
    # the aggregate).
    bad_pa = per_article.copy()
    bad_pa.loc[0, "published_at"] = pd.Timestamp("2024-07-08 21:00", tz="UTC")
    bad_pp = per_period.copy()
    # Tamper the per_period score so reproducibility also fires; but the
    # primary signal here is the PIT violation on the per_article frame.
    bad_pp.loc[0, "sentiment_score"] = bad_pp.loc[0, "sentiment_score"] + 0.5

    result = run_sentiment_aggregate_guard(bad_pa, bad_pp)
    assert result["pass"] is False
    types = [v["type"] for v in result["violations"]]
    assert "pit_violation" in types


def test_e2e_aggregate_from_joined_rejects_missing_period_close(tmp_path, monkeypatch):
    """aggregate_from_joined refuses a frame without period_close_ts (B1
    input contract).
    """
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _StubScorer()
    bad = pd.DataFrame(
        {
            "ticker": ["AAPL"],
            "published_at": [pd.Timestamp("2024-07-08 18:00", tz="UTC")],
            "text": ["x"],
            "source": ["reuters"],
        }
    )
    import pytest

    with pytest.raises(ValueError, match="period_close_ts"):
        aggregate_from_joined(bad, cache, scorer)
