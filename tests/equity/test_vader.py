"""
@module: tests.equity.test_vader
@depends: equity.sentiment.vader
@exports:
@data_flow: VADERScorer construction + lazy-load + classify contract

Real-VADER smoke test is gated behind ``vaderSentiment`` importability
(the [sentiment] extra is NOT installed by default). Default runs exercise
only construction + the lazy-load error path.
"""

from __future__ import annotations

import pytest

from equity.sentiment.vader import VADERScorer


def _vader_available() -> bool:
    try:
        import vaderSentiment  # noqa: F401

        return True
    except ImportError:
        return False


def test_vader_metadata():
    s = VADERScorer()
    assert s.model_name == "vader"
    # ``model_revision`` is the package pin string (a constant); it is part
    # of the cache key surface but does not correspond to an HF revision.
    assert isinstance(s.model_revision, str)
    assert s.model_revision


def test_vader_lazy_analyzer_is_none_until_classify():
    s = VADERScorer()
    assert s._analyzer is None  # noqa: SLF001 - intentional lazy-load check


def test_vader_classify_batch_empty_returns_empty():
    s = VADERScorer()
    assert s.classify_batch([]) == []


def test_vader_to_score_maps_polarity_to_canonical():
    """``_to_score`` maps VADER's ``{pos, neu, neg, compound}`` to
    :class:`SentimentScore` with ``score = pos - neg`` (NOT compound).
    Tested directly so the mapping is verified without requiring
    ``vaderSentiment`` installed.
    """
    s = VADERScorer()
    polarity = {"pos": 0.5, "neg": 0.3, "neu": 0.2, "compound": 0.8}
    score = s._to_score(polarity)  # noqa: SLF001 - direct helper test
    assert score.pos == pytest.approx(0.5)
    assert score.neg == pytest.approx(0.3)
    assert score.neu == pytest.approx(0.2)
    # score = pos - neg, NOT compound.
    assert score.score == pytest.approx(0.2)
    assert score.score != pytest.approx(0.8)


def test_vader_to_score_handles_missing_keys():
    s = VADERScorer()
    score = s._to_score({})  # noqa: SLF001
    assert score.pos == 0.0
    assert score.neg == 0.0
    assert score.neu == 0.0
    assert score.score == 0.0


def test_vader_load_analyzer_raises_without_vadersentiment():
    """When ``vaderSentiment`` is not installed, the lazy ``_load_analyzer``
    raises ``ImportError`` with a helpful install hint.
    """
    if _vader_available():
        pytest.skip("vaderSentiment installed -- ImportError path not exercised")
    s = VADERScorer()
    with pytest.raises(ImportError, match="sentiment"):
        s.classify("anything")


def test_vader_load_analyzer_raises_on_batch_without_vadersentiment():
    if _vader_available():
        pytest.skip("vaderSentiment installed -- ImportError path not exercised")
    s = VADERScorer()
    with pytest.raises(ImportError, match="sentiment"):
        s.classify_batch(["anything"])


@pytest.mark.skipif(
    not _vader_available(),
    reason="Real VADER smoke test requires the [sentiment] extra (vaderSentiment)",
)
def test_vader_live_classify_returns_canonical_dict():
    s = VADERScorer()
    out = s.classify("Apple reports record quarterly revenue, beating expectations.")
    assert set(out.keys()) == {"pos", "neg", "neu", "score"}
    # VADER polarity scores sum to 1 by construction.
    assert out["pos"] + out["neg"] + out["neu"] == pytest.approx(1.0, abs=1e-9)
    # ``score = pos - neg`` (NOT compound).
    assert out["score"] == pytest.approx(out["pos"] - out["neg"], abs=1e-9)
