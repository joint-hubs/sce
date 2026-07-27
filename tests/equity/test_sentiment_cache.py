"""
@module: tests.equity.test_sentiment_cache
@depends: equity.sentiment.cache, equity.sentiment.base
@exports:
@data_flow: StubScorer + synthetic articles -> SentimentCache.score_articles

Cache idempotency is the S2.2 DoD: re-running ``score_articles`` on the same
articles with the same model + revision MUST make ZERO scorer forward
passes. The StubScorer counts ``classify_batch`` calls; the second run must
not increment the counter.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from equity.sentiment.base import SentimentScorer
from equity.sentiment.cache import (
    CACHE_FILE_NAME,
    META_FILE_NAME,
    SentimentCache,
    compute_article_key,
)


class _CountingStubScorer:
    """Stub scorer that counts ``classify_batch`` invocations and returns
    deterministic scores. Satisfies the :class:`SentimentScorer` Protocol.
    """

    def __init__(self) -> None:
        self.batch_calls = 0
        self.classify_calls = 0

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def model_revision(self) -> str:
        return "v1"

    def classify(self, text: str) -> dict[str, float]:
        self.classify_calls += 1
        # Deterministic pseudo-score: hash(text) mod 1000 / 1000.
        h = (hash(text) & 0xFFFF) / 0xFFFF
        pos = 0.5 * h + 0.25
        neg = 0.25 * (1.0 - h)
        neu = 1.0 - pos - neg
        return {"pos": pos, "neg": neg, "neu": neu, "score": pos - neg}

    def classify_batch(self, texts: list[str]) -> list[dict[str, float]]:
        self.batch_calls += 1
        return [self._score_one(t) for t in texts]

    def _score_one(self, text: str) -> dict[str, float]:
        # NOTE: deliberately NOT incrementing classify_calls here (batch
        # path is what the cache uses). ``classify`` increments separately.
        h = (hash(text) & 0xFFFF) / 0xFFFF
        pos = 0.5 * h + 0.25
        neg = 0.25 * (1.0 - h)
        neu = 1.0 - pos - neg
        return {"pos": pos, "neg": neg, "neu": neu, "score": pos - neg}


def _sample_articles(n: int = 4) -> pd.DataFrame:
    pub = pd.date_range("2024-07-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "NVDA", "AAPL"][:n],
            "published_at": pub,
            "text": [f"article {i} for ticker" for i in range(n)],
            "source": ["reuters"] * n,
        }
    )


def test_stub_scorer_satisfies_protocol():
    scorer = _CountingStubScorer()
    assert isinstance(scorer, SentimentScorer)


def test_compute_article_key_is_deterministic():
    k1 = compute_article_key("hello", "stub", "v1")
    k2 = compute_article_key("hello", "stub", "v1")
    assert k1 == k2
    # Different revision -> different key (cache invalidation on fine-tune).
    k3 = compute_article_key("hello", "stub", "v2")
    assert k1 != k3
    # Different text -> different key.
    k4 = compute_article_key("world", "stub", "v1")
    assert k1 != k4


def test_compute_article_key_length_prefix_disambiguation():
    # Without length prefixes, ``("ab","c")`` and ``("a","bc")`` would hash
    # to the same naive-concatenation key. Length-prefixed hashing avoids
    # this collision class.
    k1 = compute_article_key("ab", "c", "")
    k2 = compute_article_key("a", "bc", "")
    assert k1 != k2


def test_score_articles_writes_cache_and_meta(tmp_path, monkeypatch):
    # Point PROJECT_ROOT at tmp_path so the containment guard permits
    # writing under tmp_path (which is outside the real repo root). Mirrors
    # the S1 fetch_prices tests.
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _CountingStubScorer()
    articles = _sample_articles()
    out = cache.score_articles(articles, scorer)
    assert (tmp_path / "cache" / CACHE_FILE_NAME).exists()
    assert (tmp_path / "cache" / META_FILE_NAME).exists()
    assert len(out) == len(articles)
    assert scorer.batch_calls == 1


def test_score_articles_idempotent_zero_forward_passes(tmp_path, monkeypatch):
    """DoD S2.2: re-running the cache on the same articles must make ZERO
    model forward passes.
    """
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _CountingStubScorer()
    articles = _sample_articles()

    # Cold run: 1 batch call.
    cache.score_articles(articles, scorer)
    cold_calls = scorer.batch_calls
    assert cold_calls == 1

    # Warm run: 0 additional batch calls.
    cache.score_articles(articles, scorer)
    assert scorer.batch_calls == cold_calls


def test_score_articles_warm_run_under_5pct_cold_latency(tmp_path, monkeypatch):
    """DoD S2.2: warm-run latency < 5% of cold on a stub scorer.

    The DoD's ratio is meaningful only when the cold path is dominated by
    the scorer's forward pass (a real FinBERT call takes seconds). With a
    pure stub scorer the parquet I/O dominates both paths and the ratio is
    noise. We simulate a real scorer by sleeping 50ms per batch call -- the
    warm path skips the call entirely and is dominated by the cache read,
    which is well under 5% of the cold path.
    """
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)

    class _SlowStubScorer(_CountingStubScorer):
        def classify_batch(self, texts: list[str]) -> list[dict[str, float]]:
            time.sleep(1.0)  # simulate a real FinBERT forward pass
            return super().classify_batch(texts)

    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _SlowStubScorer()
    articles = _sample_articles()

    t0 = time.perf_counter()
    cache.score_articles(articles, scorer)
    cold = time.perf_counter() - t0

    t1 = time.perf_counter()
    cache.score_articles(articles, scorer)
    warm = time.perf_counter() - t1

    assert scorer.batch_calls == 1, "warm run must not invoke the scorer"
    assert warm < cold * 0.05, f"warm latency {warm*1e3:.3f}ms not < 5% of cold {cold*1e3:.3f}ms"


def test_score_articles_partial_cache_hit(tmp_path, monkeypatch):
    """Adding a new article triggers exactly ONE batch call for the miss."""
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _CountingStubScorer()
    articles = _sample_articles(4)
    cache.score_articles(articles, scorer)
    assert scorer.batch_calls == 1

    # Add 2 new articles.
    new_articles = pd.concat(
        [
            articles,
            pd.DataFrame(
                {
                    "ticker": ["GOOG", "TSLA"],
                    "published_at": pd.date_range("2024-07-10", periods=2, freq="h", tz="UTC"),
                    "text": ["new article 1", "new article 2"],
                    "source": ["bloomberg", "bloomberg"],
                }
            ),
        ],
        ignore_index=True,
    )
    cache.score_articles(new_articles, scorer)
    # Second run: 1 batch call for the 2 new misses.
    assert scorer.batch_calls == 2


def test_score_articles_revision_change_invalidates_cache(tmp_path, monkeypatch):
    """Changing the scorer's revision must re-score every article (cache
    key includes revision).
    """
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))

    # Use two distinct scorer instances with different revisions.
    class _V2(_CountingStubScorer):
        @property
        def model_revision(self) -> str:
            return "v2"

    scorer_v1 = _CountingStubScorer()
    scorer_v2 = _V2()
    articles = _sample_articles()

    cache.score_articles(articles, scorer_v1)
    assert scorer_v1.batch_calls == 1

    # Same articles, new revision -> all keys miss -> 1 batch call.
    cache.score_articles(articles, scorer_v2)
    assert scorer_v2.batch_calls == 1


def test_score_articles_meta_records_scorer_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _CountingStubScorer()
    cache.score_articles(_sample_articles(), scorer)

    import json

    meta = json.loads((tmp_path / "cache" / META_FILE_NAME).read_text())
    assert meta["model_name"] == "stub"
    assert meta["model_revision"] == "v1"
    assert meta["scorer_class"] == "_CountingStubScorer"
    assert meta["n_articles"] == 4
    assert "cached_at_utc" in meta
    assert "content_hash" in meta


def test_score_articles_rejects_missing_text_column(tmp_path, monkeypatch):
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _CountingStubScorer()
    bad = pd.DataFrame({"ticker": ["AAPL"], "published_at": [pd.Timestamp("2024-07-01", tz="UTC")]})
    with pytest.raises(ValueError, match="missing required column 'text'"):
        cache.score_articles(bad, scorer)


def test_cache_dir_outside_project_root_rejected(tmp_path):
    # tmp_path is NOT under PROJECT_ROOT -- the containment guard should
    # refuse it ONLY for absolute paths. Relative paths are joined to
    # PROJECT_ROOT, so they always pass. Use an absolute path outside.
    with pytest.raises(ValueError, match="outside PROJECT_ROOT"):
        SentimentCache(cache_dir=str(tmp_path / "outside"))


def test_score_articles_rejects_missing_ticker_column(tmp_path, monkeypatch):
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _CountingStubScorer()
    bad = pd.DataFrame(
        {
            "text": ["x"],
            "published_at": [pd.Timestamp("2024-07-01", tz="UTC")],
        }
    )
    with pytest.raises(ValueError, match="missing required column 'ticker'"):
        cache.score_articles(bad, scorer)


def test_score_articles_rejects_missing_published_at_column(tmp_path, monkeypatch):
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _CountingStubScorer()
    bad = pd.DataFrame(
        {
            "text": ["x"],
            "ticker": ["AAPL"],
        }
    )
    with pytest.raises(ValueError, match="missing required column 'published_at'"):
        cache.score_articles(bad, scorer)


def test_score_articles_raises_on_scorer_length_mismatch(tmp_path, monkeypatch):
    """If the scorer returns the wrong number of scores, the cache must
    raise ``RuntimeError`` (defensive -- a misbehaving scorer must not
    silently misalign cached scores).
    """

    class _BadScorer(_CountingStubScorer):
        def classify_batch(self, texts: list[str]) -> list[dict[str, float]]:
            self.batch_calls += 1
            # Return ONE score regardless of input length.
            return [{"pos": 0.5, "neg": 0.3, "neu": 0.2, "score": 0.2}]

    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    articles = _sample_articles(n=3)
    with pytest.raises(RuntimeError, match="length mismatch"):
        cache.score_articles(articles, _BadScorer())


def test_get_meta_returns_none_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    assert cache.get_meta() is None


def test_get_meta_returns_parsed_dict_after_write(tmp_path, monkeypatch):
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _CountingStubScorer()
    cache.score_articles(_sample_articles(), scorer)
    meta = cache.get_meta()
    assert meta is not None
    assert meta["model_name"] == "stub"
    assert meta["n_articles"] == 4


def test_get_meta_returns_none_on_corrupt_json(tmp_path, monkeypatch):
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    cache.cache_dir.mkdir(parents=True, exist_ok=True)
    cache.meta_path.write_text("{not valid json")
    assert cache.get_meta() is None


def test_score_articles_empty_articles_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("equity.sentiment.cache.PROJECT_ROOT", tmp_path)
    cache = SentimentCache(cache_dir=str(tmp_path / "cache"))
    scorer = _CountingStubScorer()
    empty = pd.DataFrame(columns=["ticker", "published_at", "text", "source"])
    out = cache.score_articles(empty, scorer)
    assert len(out) == 0
    assert scorer.batch_calls == 0
