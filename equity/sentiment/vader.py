"""
@module: equity.sentiment.vader
@depends: vaderSentiment (lazy)
@exports: VADERScorer
@paper_ref: Hutto & Gilbert (2014), VADER
@data_flow: text -> SentimentIntensityAnalyzer.polarity_scores -> {pos,neg,neu,score}

VADER fallback scorer. Implements the SAME :class:`SentimentScorer` contract
as :class:`FinBERTScorer` so the cache / aggregation layers are agnostic to
which scorer produced the scores.

Output contract
---------------
VADER's ``polarity_scores(text)`` returns
``{"neg", "neu", "pos", "compound"}``. We map to the canonical
``{"pos", "neg", "neu", "score"}`` where ``score = pos - neg`` (NOT
``compound``). ``compound`` is a lexicon-intensity-weighted score in
``[-1, 1]`` that is NOT comparable to FinBERT's softmax-derived ``pos - neg``;
normalizing both scorers to ``pos - neg`` keeps the aggregation semantics
identical regardless of the active scorer.
"""

from __future__ import annotations

from typing import Any

from equity.sentiment.base import SentimentScore, make_score

_DEFAULT_REVISION = "vaderSentiment>=3.3.2"


class VADERScorer:
    """VADER lexicon-based sentiment scorer (lazy import).

    VADER is rule-based (no model weights), so there is no HF revision to
    pin. :attr:`model_revision` returns the package pin string (a constant)
    purely to satisfy the :class:`SentimentScorer` Protocol's cache-key
    contract -- bumping the constant invalidates cached scores if a future
    VADER release changes the lexicon.
    """

    def __init__(self, revision: str = _DEFAULT_REVISION) -> None:
        self._revision = revision
        self._analyzer: Any | None = None  # lazy

    # -- Properties (cache key surface) -----------------------------------

    @property
    def model_name(self) -> str:
        return "vader"

    @property
    def model_revision(self) -> str:
        return self._revision

    # -- Lazy analyzer load -----------------------------------------------

    def _load_analyzer(self) -> Any:
        """Lazy-import ``vaderSentiment`` and build the analyzer on first use.

        Deferring the import keeps ``import equity.sentiment`` light; the
        scorer is also used as a fallback (FinBERT is the default), so most
        runs never touch this path.
        """
        if self._analyzer is not None:
            return self._analyzer
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:  # pragma: no cover - exercised in live test
            raise ImportError(
                "VADERScorer requires the 'vaderSentiment' package. Install with: "
                "pip install -e '.[sentiment]'"
            ) from exc
        self._analyzer = SentimentIntensityAnalyzer()
        return self._analyzer

    # -- Scoring ----------------------------------------------------------

    def _to_score(self, polarity: dict[str, float]) -> SentimentScore:
        """Map VADER's ``{pos, neu, neg, compound}`` to :class:`SentimentScore`."""
        pos = float(polarity.get("pos", 0.0))
        neg = float(polarity.get("neg", 0.0))
        neu = float(polarity.get("neu", 0.0))
        return make_score(pos=pos, neg=neg, neu=neu)

    def classify(self, text: str) -> dict[str, float]:
        """Score a single text."""
        analyzer = self._load_analyzer()
        polarity = analyzer.polarity_scores(text)
        score = self._to_score(polarity)
        return {
            "pos": score.pos,
            "neg": score.neg,
            "neu": score.neu,
            "score": score.score,
        }

    def classify_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Score a batch of texts. VADER has no native batching; this is a
        simple loop. Returns dicts in input order.
        """
        if not texts:
            return []
        return [self.classify(t) for t in texts]


__all__ = ["VADERScorer"]
