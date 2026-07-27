"""
@module: equity.sentiment.finbert
@depends: transformers, torch (lazy)
@exports: FinBERTScorer
@paper_ref: ProsusAI/finbert
@data_flow: text -> HF transformers pipeline -> softmax {pos,neg,neu} -> score

FinBERT-based sentiment scorer. The HuggingFace ``transformers`` pipeline is
**lazy-loaded** on the first ``classify`` / ``classify_batch`` call so
``import equity.sentiment`` is light and tests can exercise the cache /
aggregation layer without ``torch`` installed (use a stub scorer instead).

The default model is ``ProsusAI/finbert`` at a pinned revision. The revision
is part of the cache key (see :mod:`equity.sentiment.cache`), so a fine-tune
or model update invalidates cached scores cleanly.

Output contract
---------------
The pipeline returns ``[{"positive": p, "negative": n, "neutral": t}]``;
``classify`` normalizes to ``{"pos": p, "neg": n, "neu": t, "score": p - n}``
(see :mod:`equity.sentiment.base`). ``pos + neg + neu`` sums to 1 up to
float32 rounding; the schema's ``_assert_probs_sum_to_one`` tolerance absorbs
this.
"""

from __future__ import annotations

from typing import Any

from equity.sentiment.base import SentimentScore, make_score

# Default HF model + revision. ``revision="main"`` is the moving default; a
# TODO below marks the recommended pin. The revision is part of the cache
# key, so bumping it (e.g. to a specific commit SHA after a model audit)
# invalidates cached scores cleanly.
_DEFAULT_MODEL = "ProsusAI/finbert"
_DEFAULT_REVISION = "main"
# TODO(FOC-49): pin ``_DEFAULT_REVISION`` to a specific commit SHA after
# auditing the ProsusAI/finbert history, so cached scores are reproducible
# against an immutable model snapshot rather than the moving ``main`` ref.


class FinBERTScorer:
    """FinBERT sentiment scorer (lazy transformers pipeline).

    Parameters
    ----------
    model_name:
        HF model id (default ``"ProsusAI/finbert"``).
    batch_size:
        Number of texts per forward pass in :meth:`classify_batch`. The
        pipeline batches internally; this is the chunk size for very large
        inputs to bound peak memory.
    device:
        Torch device (``"cpu"``, ``"cuda"``, ``None`` for auto). ``None``
        lets the pipeline pick (CPU on a CPU-only box). Do NOT force CUDA --
        CPU torch is fine for the seed workload.
    revision:
        HF revision (commit SHA or branch). Default ``"main"``; see
        :data:`_DEFAULT_REVISION` TODO. The revision is exposed via
        :attr:`model_revision` and used in the cache key.
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = 8,
        device: str | None = None,
        revision: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._batch_size = int(batch_size)
        self._device = device
        self._revision = revision or _DEFAULT_REVISION
        self._pipeline: Any | None = None  # lazy

    # -- Properties (cache key surface) -----------------------------------

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_revision(self) -> str:
        return self._revision

    @property
    def batch_size(self) -> int:
        return self._batch_size

    # -- Lazy pipeline load -----------------------------------------------

    def _load_pipeline(self) -> Any:
        """Lazy-load the HF sentiment-analysis pipeline on first use.

        Importing ``transformers`` and ``torch`` is expensive (multi-second
        import, large memory footprint); defer it until the first
        ``classify`` call so ``import equity.sentiment`` stays light and
        tests can use a stub scorer without the deps installed.
        """
        if self._pipeline is not None:
            return self._pipeline
        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError as exc:  # pragma: no cover - exercised in live test
            raise ImportError(
                "FinBERTScorer requires the 'transformers' extra. Install with: "
                "pip install -e '.[sentiment]'"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, revision=self._revision)
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name, revision=self._revision
        )
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=self._device,
            top_k=3,  # return all 3 logits (positive, negative, neutral)
            return_all_scores=False,
        )
        return self._pipeline

    # -- Scoring ----------------------------------------------------------

    def _normalize(self, raw: list[dict]) -> SentimentScore:
        """Map the pipeline's ``[{label, score}, ...]`` output to the
        canonical :class:`SentimentScore` (``pos - neg``).
        """
        by_label = {item["label"].lower(): float(item["score"]) for item in raw}
        pos = by_label.get("positive", 0.0)
        neg = by_label.get("negative", 0.0)
        neu = by_label.get("neutral", 0.0)
        return make_score(pos=pos, neg=neg, neu=neu)

    def classify(self, text: str) -> dict[str, float]:
        """Score a single text. Lazy-loads the pipeline on first call."""
        pipe = self._load_pipeline()
        raw = pipe(text)
        # ``top_k=3`` returns ``[[{label, score}, ...]]`` for a single string
        # input; unpack the outer list.
        if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
            raw = raw[0]
        score = self._normalize(raw)
        return {
            "pos": score.pos,
            "neg": score.neg,
            "neu": score.neu,
            "score": score.score,
        }

    def classify_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Score a batch of texts. Chunks by ``batch_size`` to bound peak
        memory. Returns dicts in input order.
        """
        if not texts:
            return []
        pipe = self._load_pipeline()
        results: list[dict[str, float]] = []
        for start in range(0, len(texts), self._batch_size):
            chunk = texts[start : start + self._batch_size]
            raw_batch = pipe(chunk)
            # ``top_k=3`` returns a list-of-lists when given a list input.
            for raw in raw_batch:
                if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
                    raw = raw[0]
                if isinstance(raw, list):
                    # already a list of {label, score}
                    pass
                score = self._normalize(raw if isinstance(raw, list) else [raw])
                results.append(
                    {
                        "pos": score.pos,
                        "neg": score.neg,
                        "neu": score.neu,
                        "score": score.score,
                    }
                )
        return results


__all__ = ["FinBERTScorer"]
