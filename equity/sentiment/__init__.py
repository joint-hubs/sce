"""
@module: equity.sentiment
@depends: typing
@exports: SentimentScorer, FinBERTScorer, VADERScorer, get_scorer,
          SentimentCache, aggregate_per_period, aggregate_market_wide
@paper_ref: N/A
@data_flow: text -> SentimentScorer -> per-article cache -> per-(ticker,period) aggregate

S2 sentiment layer. Re-exports are LAZY so ``import equity.sentiment`` is
light (no torch / transformers / vaderSentiment at import time). Mirror S1's
``equity.diagnostics`` lazy-export style.
"""

from __future__ import annotations


def __getattr__(name: str):  # pragma: no cover - thin re-export
    if name in ("SentimentScore", "SentimentScorer", "score_to_dict", "make_score", "get_scorer"):
        from equity.sentiment import base

        return getattr(base, name)
    if name == "FinBERTScorer":
        from equity.sentiment.finbert import FinBERTScorer

        return FinBERTScorer
    if name == "VADERScorer":
        from equity.sentiment.vader import VADERScorer

        return VADERScorer
    if name in ("SentimentCache", "compute_article_key"):
        from equity.sentiment import cache

        return getattr(cache, name)
    if name in (
        "aggregate_per_period",
        "aggregate_market_wide",
        "aggregate_from_joined",
        "build_market_wide_per_article",
        "write_sentiment_per_period",
        "write_market_wide",
        "DEFAULT_DECAY_TIME_CONST_DAYS",
    ):
        from equity.sentiment import aggregate

        return getattr(aggregate, name)
    raise AttributeError(f"module 'equity.sentiment' has no attribute {name!r}")


__all__ = [
    "SentimentScore",
    "SentimentScorer",
    "score_to_dict",
    "make_score",
    "get_scorer",
    "FinBERTScorer",
    "VADERScorer",
    "SentimentCache",
    "compute_article_key",
    "aggregate_per_period",
    "aggregate_market_wide",
    "aggregate_from_joined",
    "build_market_wide_per_article",
    "write_sentiment_per_period",
    "write_market_wide",
    "DEFAULT_DECAY_TIME_CONST_DAYS",
]
