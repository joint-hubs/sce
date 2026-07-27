"""
@module: tests.equity.test_finbert
@depends: equity.sentiment.finbert
@exports:
@data_flow: FinBERTScorer construction + lazy-load + classify contract

Real-FinBERT smoke test is gated behind ``SCE_EQUITY_LIVE_TEST=1`` AND
``transformers``/``torch`` importability (the [sentiment] extra is NOT
installed by default). Default runs exercise only construction + the lazy-
load error path.
"""

from __future__ import annotations

import os

import pytest

from equity.sentiment.finbert import FinBERTScorer


def _hf_available() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def test_finbert_default_metadata():
    s = FinBERTScorer()
    assert s.model_name == "ProsusAI/finbert"
    assert s.model_revision == "main"
    assert s.batch_size == 8


def test_finbert_custom_revision():
    s = FinBERTScorer(revision="abc123")
    assert s.model_revision == "abc123"


def test_finbert_lazy_pipeline_is_none_until_classify():
    s = FinBERTScorer()
    assert s._pipeline is None  # noqa: SLF001 - intentional lazy-load check


def test_finbert_normalize_maps_labels_to_canonical_dict():
    """``_normalize`` maps the HF pipeline's ``[{label, score}, ...]`` output
    to the canonical ``SentimentScore`` (``pos - neg``). Tested directly so
    the contract is verified without requiring ``torch`` / ``transformers``.
    """
    s = FinBERTScorer()
    raw = [
        {"label": "positive", "score": 0.7},
        {"label": "negative", "score": 0.2},
        {"label": "neutral", "score": 0.1},
    ]
    score = s._normalize(raw)  # noqa: SLF001 - direct helper test
    assert score.pos == pytest.approx(0.7)
    assert score.neg == pytest.approx(0.2)
    assert score.neu == pytest.approx(0.1)
    assert score.score == pytest.approx(0.5)


def test_finbert_normalize_handles_missing_labels():
    """Missing labels default to 0.0 (defensive -- a malformed pipeline
    output should not crash the scorer).
    """
    s = FinBERTScorer()
    raw = [{"label": "positive", "score": 1.0}]
    score = s._normalize(raw)  # noqa: SLF001
    assert score.pos == pytest.approx(1.0)
    assert score.neg == pytest.approx(0.0)
    assert score.neu == pytest.approx(0.0)


def test_finbert_classify_batch_empty_returns_empty():
    s = FinBERTScorer()
    assert s.classify_batch([]) == []


def test_finbert_load_pipeline_raises_without_transformers():
    """When ``transformers`` / ``torch`` are not installed, the lazy
    ``_load_pipeline`` raises ``ImportError`` with a helpful install hint.
    Exercised only when the [sentiment] extra is NOT installed.
    """
    if _hf_available():
        pytest.skip("transformers/torch installed -- ImportError path not exercised")
    s = FinBERTScorer()
    with pytest.raises(ImportError, match="sentiment"):
        s.classify("anything")


def test_finbert_load_pipeline_raises_on_batch_without_transformers():
    """Same ImportError path via ``classify_batch``."""
    if _hf_available():
        pytest.skip("transformers/torch installed -- ImportError path not exercised")
    s = FinBERTScorer()
    with pytest.raises(ImportError, match="sentiment"):
        s.classify_batch(["anything"])


@pytest.mark.skipif(
    not (_hf_available() and os.environ.get("SCE_EQUITY_LIVE_TEST") == "1"),
    reason="Real FinBERT smoke test requires SCE_EQUITY_LIVE_TEST=1 and the [sentiment] extra",
)
def test_finbert_live_classify_returns_canonical_dict():
    # Local import so the module-level skipif gates the actual HF download.
    s = FinBERTScorer()
    out = s.classify("Apple reports record quarterly revenue.")
    assert set(out.keys()) == {"pos", "neg", "neu", "score"}
    assert out["pos"] + out["neg"] + out["neu"] == pytest.approx(1.0, abs=1e-5)
    assert out["score"] == pytest.approx(out["pos"] - out["neg"], abs=1e-9)
