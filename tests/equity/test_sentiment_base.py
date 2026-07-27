"""
@module: tests.equity.test_sentiment_base
@depends: equity.sentiment.base
@exports:
@data_flow: SentimentScore / make_score / score_to_dict / get_scorer
"""

from __future__ import annotations

import pytest

from equity.sentiment.base import (
    SentimentScorer,
    get_scorer,
    make_score,
    score_to_dict,
)


def test_make_score_computes_pos_minus_neg():
    s = make_score(pos=0.7, neg=0.2, neu=0.1)
    assert s.score == pytest.approx(0.5)


def test_score_to_dict_canonical_keys():
    s = make_score(pos=0.5, neg=0.3, neu=0.2)
    d = score_to_dict(s)
    assert set(d.keys()) == {"pos", "neg", "neu", "score"}
    assert d["score"] == pytest.approx(0.2)


def test_sentiment_score_is_frozen_dataclass():
    s = make_score(pos=0.5, neg=0.3, neu=0.2)
    with pytest.raises(Exception):
        s.pos = 0.9  # type: ignore[misc]


class _StubScorer:
    """Minimal scorer for Protocol compliance checks."""

    def __init__(self) -> None:
        self._calls = 0

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def model_revision(self) -> str:
        return "v1"

    def classify(self, text: str) -> dict[str, float]:
        self._calls += 1
        return {"pos": 0.5, "neg": 0.3, "neu": 0.2, "score": 0.2}

    def classify_batch(self, texts: list[str]) -> list[dict[str, float]]:
        self._calls += 1
        return [self.classify(t) for t in texts]  # double-counts, but ok for shape


def test_stub_scorer_satisfies_protocol():
    scorer = _StubScorer()
    assert isinstance(scorer, SentimentScorer)


def test_get_scorer_unknown_name_raises():
    with pytest.raises(ValueError, match="Unknown sentiment scorer"):
        get_scorer("not-a-scorer")


def test_get_scorer_env_override(monkeypatch):
    # The default scorer is FinBERT; we can't construct it without torch, so
    # patch the FinBERTScorer import target.
    import equity.sentiment.base as base

    class _FakeFin:
        pass

    class _FakeVader:
        pass

    monkeypatch.setattr("equity.sentiment.finbert.FinBERTScorer", _FakeFin, raising=False)
    import equity.sentiment.finbert as finbert

    monkeypatch.setattr(finbert, "FinBERTScorer", _FakeFin, raising=False)
    import equity.sentiment.vader as vader

    monkeypatch.setattr(vader, "VADERScorer", _FakeVader, raising=False)

    monkeypatch.delenv("SENTIMENT_SCORER", raising=False)
    out = base.get_scorer(None)
    assert isinstance(out, _FakeFin)

    monkeypatch.setenv("SENTIMENT_SCORER", "vader")
    out = base.get_scorer(None)
    assert isinstance(out, _FakeVader)

    monkeypatch.delenv("SENTIMENT_SCORER", raising=False)
    out = base.get_scorer("vader")
    assert isinstance(out, _FakeVader)


def test_get_scorer_default_is_finbert(monkeypatch):
    import equity.sentiment.finbert as finbert

    class _FakeFin:
        pass

    monkeypatch.setattr(finbert, "FinBERTScorer", _FakeFin, raising=False)
    monkeypatch.delenv("SENTIMENT_SCORER", raising=False)
    out = get_scorer()
    assert isinstance(out, _FakeFin)
