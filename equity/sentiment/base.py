"""
@module: equity.sentiment.base
@depends: typing, os
@exports: SentimentScore, SentimentScorer, score_to_dict, get_scorer
@paper_ref: N/A
@data_flow: text -> SentimentScorer.classify -> {pos, neg, neu, score}

The sentiment-scorer Protocol shared by :class:`FinBERTScorer` (S2.1) and
:class:`VADERScorer` (S2.4 fallback). Every scorer returns a dict shaped::

    {"pos": float, "neg": float, "neu": float, "score": float}

where ``score = pos - neg`` (NOT compound -- VADER's ``compound`` is a
lexicon-intensity-weighted score that is NOT comparable to FinBERT's
softmax-derived ``pos - neg``; the contract normalizes both scorers to the
same definition). ``pos + neg + neu`` is required to sum to 1 (within
tolerance; see :mod:`equity.sentiment.schema`).

Cache keying
------------
The cache (see :mod:`equity.sentiment.cache`) keys on
``sha256(text + model_name + model_revision)``. Every scorer MUST expose
``model_name`` and ``model_revision`` properties so a fine-tune / model
update invalidates cached scores cleanly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SentimentScore:
    """Normalized sentiment probabilities + the ``pos - neg`` score.

    Attributes
    ----------
    pos, neg, neu:
        Probabilities in ``[0, 1]`` summing to 1 (within tolerance).
    score:
        ``pos - neg``. Range ``[-1, 1]``. This is the canonical scalar used
        by the per-(ticker, period) aggregation (S2.3).
    """

    pos: float
    neg: float
    neu: float
    score: float


def score_to_dict(score: SentimentScore) -> dict[str, float]:
    """Convert a :class:`SentimentScore` to the canonical dict contract."""
    return {
        "pos": float(score.pos),
        "neg": float(score.neg),
        "neu": float(score.neu),
        "score": float(score.score),
    }


def make_score(pos: float, neg: float, neu: float) -> SentimentScore:
    """Build a :class:`SentimentScore` from probabilities, computing
    ``score = pos - neg``. The caller is responsible for ensuring
    ``pos + neg + neu ≈ 1`` (validated downstream by the schema helpers).
    """
    return SentimentScore(pos=float(pos), neg=float(neg), neu=float(neu), score=float(pos - neg))


@runtime_checkable
class SentimentScorer(Protocol):
    """Protocol every sentiment scorer implements.

    ``classify(text)`` returns the canonical dict (see module docstring).
    ``classify_batch(texts)`` returns a list of dicts in input order (for
    vectorized scorers like FinBERT this avoids per-article pipeline
    overhead). Both methods must be deterministic given the same input text
    and the same ``model_name`` / ``model_revision`` (the cache relies on
    this for idempotency).
    """

    @property
    def model_name(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    def classify(self, text: str) -> dict[str, float]: ...

    def classify_batch(self, texts: list[str]) -> list[dict[str, float]]: ...


def get_scorer(name: str | None = None) -> SentimentScorer:
    """Factory returning a scorer by name, with environment override.

    Resolution order:
    1. Explicit ``name`` argument (``"finbert"`` or ``"vader"``).
    2. ``SENTIMENT_SCORER`` env var (``"finbert"`` / ``"vader"``).
    3. Default: ``"finbert"``.

    Lazy-imports the scorer implementation so ``import equity.sentiment`` is
    light (no torch / transformers / vaderSentiment at import time).

    Parameters
    ----------
    name:
        Optional scorer name. When ``None``, falls back to the
        ``SENTIMENT_SCORER`` env var, then to ``"finbert"``.

    Returns
    -------
    SentimentScorer
        A concrete scorer instance.

    Raises
    ------
    ValueError
        If ``name`` (or the env var) is not ``"finbert"`` or ``"vader"``.
    """
    resolved = name or os.environ.get("SENTIMENT_SCORER") or "finbert"
    resolved = resolved.strip().lower()
    if resolved == "finbert":
        from equity.sentiment.finbert import FinBERTScorer

        return FinBERTScorer()
    if resolved == "vader":
        from equity.sentiment.vader import VADERScorer

        return VADERScorer()
    raise ValueError(
        f"Unknown sentiment scorer '{resolved}'. "
        "Expected 'finbert' or 'vader' (or set SENTIMENT_SCORER env var)."
    )


__all__ = [
    "SentimentScore",
    "SentimentScorer",
    "score_to_dict",
    "make_score",
    "get_scorer",
]
