"""
@module: equity.sentiment.cache
@depends: pandas, hashlib, json, os
@exports: SentimentCache, compute_article_key
@paper_ref: N/A
@data_flow: articles DataFrame -> scorer.classify_batch -> per-article scores cache

Idempotent per-article sentiment score cache.

Cache key
---------
Each article's score is keyed by ``sha256(text + model_name + model_revision)``
(see :func:`compute_article_key`). Including ``model_revision`` means a
FinBERT fine-tune or a VADER lexicon bump invalidates only the affected
scores cleanly -- re-running the cache against the same articles with the
same model makes ZERO model forward passes.

Storage layout
--------------
The cache is a single parquet file (``sentiment_per_article.parquet``) plus
a ``_meta.json`` sidecar mirroring S1's pattern (see
:func:`equity.data.loader._write_store_meta`). The metadata records
``model_name``, ``model_revision``, ``scorer_class``, ``cached_at_utc``,
``n_articles``, ``content_hash`` (sha256 over the canonical articles frame)
so two runs over the same articles can be compared for drift.

Idempotency contract
--------------------
Re-scoring the same articles with the same model + revision MUST make zero
model forward passes. :meth:`SentimentCache.score_articles` consults the
cache for every article_key already present and only invokes the scorer for
the missing keys. The :mod:`tests.equity.test_sentiment_cache` suite asserts
this with a stub scorer that counts ``classify_batch`` calls.

Path safety
-----------
The cache directory is resolved under ``PROJECT_ROOT`` (mirroring S1's
``_resolve_store_path`` containment guard). The directory is NOT a
``.equity_store`` (it is not a Hive-partitioned parquet dataset); the
``.equity_store`` marker pattern is intentionally NOT reused here so the S1
``_safe_rmtree`` guard does not accidentally clean the cache directory.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from equity.data.registry import PROJECT_ROOT
from equity.sentiment.base import SentimentScorer
from equity.sentiment.schema import (
    CANONICAL_PER_ARTICLE_COLUMNS,
    assert_per_article_primary_key_unique,
    validate_sentiment_per_article,
)

log = logging.getLogger(__name__)

# Default cache directory (relative to PROJECT_ROOT; gitignored -- see
# ``.gitignore``). Tests override this with a tmp_path.
DEFAULT_CACHE_DIR = ".cache/sentiment"

# File names inside the cache directory.
CACHE_FILE_NAME = "sentiment_per_article.parquet"
META_FILE_NAME = "_meta.json"


def compute_article_key(text: str, model_name: str, model_revision: str) -> str:
    """Return the sha256 cache key for a single article.

    The key is ``sha256(text + model_name + model_revision)`` (deterministic
    concatenation of the three fields with a length prefix per field so
    collisions across (text, model_name, model_revision) tuples are not
    possible via prefix overlaps -- e.g. ``("ab", "c")`` vs ``("a", "bc")``
    would hash to the same naive-concatenation key but different
    length-prefixed keys).
    """
    text_b = (text or "").encode("utf-8")
    name_b = (model_name or "").encode("utf-8")
    rev_b = (model_revision or "").encode("utf-8")
    h = hashlib.sha256()
    for field in (text_b, name_b, rev_b):
        h.update(len(field).to_bytes(8, "big"))
        h.update(field)
    return h.hexdigest()


def _resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    """Resolve the cache directory to an absolute path under PROJECT_ROOT.

    Relative paths are joined to ``PROJECT_ROOT``; absolute paths are honored
    verbatim but MUST resolve to a path inside ``PROJECT_ROOT`` (mirrors
    S1's ``_resolve_store_path`` containment guard).
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    cache_path = Path(cache_dir)
    if not cache_path.is_absolute():
        cache_path = PROJECT_ROOT / cache_path
    resolved = cache_path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        raise ValueError(
            f"Refusing to resolve sentiment cache dir {cache_path}: "
            f"resolved path {resolved} is outside PROJECT_ROOT "
            f"({PROJECT_ROOT}). Cache must live under the repo root."
        )
    return resolved


def _write_meta(
    meta_path: Path,
    *,
    model_name: str,
    model_revision: str,
    scorer_class: str,
    n_articles: int,
    content_hash: str,
) -> Path:
    """Write ``_meta.json`` atomically (mirrors S1's
    :func:`equity.data.loader._write_store_meta`).
    """
    meta: dict[str, Any] = {
        "kind": "sentiment_per_article",
        "model_name": model_name,
        "model_revision": model_revision,
        "scorer_class": scorer_class,
        "cached_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "n_articles": int(n_articles),
        "content_hash": content_hash,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix="_meta_", suffix=".tmp", dir=str(meta_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(meta, indent=2, sort_keys=True))
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, meta_path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return meta_path


def _articles_content_hash(articles: pd.DataFrame) -> str:
    """Return a deterministic sha256 over the canonical score-result frame
    (``article_key + model_name + model_revision + pos + neg + neu + score``,
    columns sorted, rows sorted) so two runs over the same articles yield
    the same hash. Mirrors S1's ``_store_content_hash``.

    Hashing the FULL score state (not just ``article_key + ticker +
    published_at``) means a corrupted probability column or a stale
    ``model_revision`` is detected as content drift (FOC-49 L8).
    """
    cols = [c for c in CANONICAL_PER_ARTICLE_COLUMNS if c in articles.columns]
    sorted_df = articles[cols].sort_values(cols).reset_index(drop=True)
    payload = sorted_df.to_json(orient="records", date_format="iso")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SentimentCache:
    """Idempotent per-article sentiment score cache.

    Parameters
    ----------
    cache_dir:
        Directory for ``sentiment_per_article.parquet`` + ``_meta.json``.
        Relative paths are joined to ``PROJECT_ROOT``; absolute paths must
        live under ``PROJECT_ROOT`` (containment guard). Defaults to
        ``.cache/sentiment``.

    Concurrency
    -----------
    The cache is NOT thread-safe and NOT process-safe. It assumes a
    single-process, single-thread access pattern. There is no lock around
    ``os.replace`` -- a concurrent miss in two processes would score the
    same articles twice and the second writer's ``os.replace`` would
    silently lose the first writer's union rows (the union is rebuilt from
    the on-disk cache + this batch's input, so rows from the OTHER process's
    concurrent batch are dropped). If you need multi-process safety, wrap
    access in an external lock or run a single scorer process.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = _resolve_cache_dir(cache_dir)
        self.cache_path = self.cache_dir / CACHE_FILE_NAME
        self.meta_path = self.cache_dir / META_FILE_NAME

    # -- I/O --------------------------------------------------------------

    def _load_cached(self) -> pd.DataFrame:
        """Load the cached per-article frame, or return an empty canonical
        frame if the cache does not exist.
        """
        if not self.cache_path.exists():
            return pd.DataFrame(columns=CANONICAL_PER_ARTICLE_COLUMNS)
        df = pd.read_parquet(self.cache_path)
        # Defensive: ensure canonical column order.
        return df[CANONICAL_PER_ARTICLE_COLUMNS]

    def _write_cache(self, df: pd.DataFrame, scorer: SentimentScorer) -> None:
        """Validate + write the cache parquet + ``_meta.json`` atomically."""
        validated = validate_sentiment_per_article(df)
        assert_per_article_primary_key_unique(validated)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # Atomic write: write to a temp file in the same dir, then os.replace.
        fd, tmp_name = tempfile.mkstemp(
            prefix="_cache_", suffix=".parquet", dir=str(self.cache_dir)
        )
        os.close(fd)
        try:
            validated.to_parquet(tmp_name, index=False)
            # 0o600: owner-only read/write (POSIX). No-op on Windows; safer
            # than world-readable 0o644 on shared POSIX hosts.
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self.cache_path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        _write_meta(
            self.meta_path,
            model_name=scorer.model_name,
            model_revision=scorer.model_revision,
            scorer_class=type(scorer).__name__,
            n_articles=int(len(validated)),
            content_hash=_articles_content_hash(validated),
        )

    # -- Public API -------------------------------------------------------

    def score_articles(
        self,
        articles: pd.DataFrame,
        scorer: SentimentScorer,
        *,
        text_col: str = "text",
        ticker_col: str = "ticker",
        published_at_col: str = "published_at",
    ) -> pd.DataFrame:
        """Score ``articles`` with ``scorer``, caching results.

        Idempotency: articles whose ``article_key`` is already in the cache
        are returned verbatim; the scorer is invoked ONLY for the missing
        keys. Re-running this method on the same articles with the same
        model + revision makes ZERO model forward passes (asserted in
        :mod:`tests.equity.test_sentiment_cache`).

        Pass-through contract (FOC-49 B1/B2)
        --------------------------------------
        The cache parquet stores ONLY the score-result surface (see
        :data:`CANONICAL_PER_ARTICLE_COLUMNS`): ``article_key, model_name,
        model_revision, pos, neg, neu, score``. The returned DataFrame is
        the score-result surface JOINED with ALL input columns EXCEPT
        ``text_col`` (so ``ticker``, ``published_at``, ``period_close_ts``,
        ``source`` etc. are passed through from the CURRENT input, never
        from the cache). This eliminates ticker drift on warm-path cache
        hits (the same text re-scored under a different ticker returns the
        new ticker from the input) and surfaces ``period_close_ts`` to
        downstream aggregation when the input carries it.

        Parameters
        ----------
        articles:
            DataFrame with at least ``text_col``, ``ticker_col``,
            ``published_at_col``. Typically ``articles.parquet`` or
            ``articles_joined.parquet`` from S1. ``ticker`` may be null
            (unresolved-ticker / market-wide articles).
        scorer:
            A :class:`SentimentScorer` (FinBERT or VADER).
        text_col, ticker_col, published_at_col:
            Column name overrides (default to the S1 articles schema).

        Returns
        -------
        pd.DataFrame
            Score-result columns (see
            :data:`CANONICAL_PER_ARTICLE_COLUMNS`) concatenated with ALL
            input columns except ``text_col`` (pass-through from the
            current input). Includes ALL cached rows for the input
            articles, not just newly-scored ones.
        """
        if text_col not in articles.columns:
            raise ValueError(f"articles frame missing required column '{text_col}'.")
        if ticker_col not in articles.columns:
            raise ValueError(f"articles frame missing required column '{ticker_col}'.")
        if published_at_col not in articles.columns:
            raise ValueError(f"articles frame missing required column '{published_at_col}'.")

        model_name = scorer.model_name
        model_revision = scorer.model_revision

        # Build article_key for every input article. NOTE: an empty/null
        # text collapses to a single cache entry (""); this is INTENDED --
        # the cache keys on (text + model_name + model_revision), and two
        # empty-text articles are indistinguishable to the scorer. Callers
        # should filter null/empty text upstream if they want per-row
        # scores for distinct articles with missing text.
        keys = [
            compute_article_key(
                str(t) if pd.notna(t) and t is not None else "", model_name, model_revision
            )
            for t in articles[text_col].tolist()
        ]

        # Build the score-result frame (cache-internal canonical surface).
        score_frame = pd.DataFrame(
            {
                "article_key": keys,
                "model_name": model_name,
                "model_revision": model_revision,
                # placeholder -- filled below
                "pos": float("nan"),
                "neg": float("nan"),
                "neu": float("nan"),
                "score": float("nan"),
            }
        )

        cached = self._load_cached()
        # Index cached by article_key for O(1) lookup.
        cached_by_key: dict[str, dict[str, float]] = {}
        if not cached.empty:
            for row in cached.itertuples(index=False):
                cached_by_key[row.article_key] = {
                    "pos": float(row.pos),
                    "neg": float(row.neg),
                    "neu": float(row.neu),
                    "score": float(row.score),
                }

        # Determine which keys need scoring.
        missing_idx = [i for i, k in enumerate(keys) if k not in cached_by_key]

        n_cache_hits = len(score_frame) - len(missing_idx)
        log.info(
            "SentimentCache: %d cache hit(s), %d miss(es) -> scorer.",
            n_cache_hits,
            len(missing_idx),
        )

        if missing_idx:
            missing_texts = [
                str(articles.iloc[i][text_col]) if pd.notna(articles.iloc[i][text_col]) else ""
                for i in missing_idx
            ]
            # Batch-score the missing texts in one forward pass (FinBERT
            # vectorizes; VADER loops). Order is preserved.
            scored = scorer.classify_batch(missing_texts)
            if len(scored) != len(missing_idx):
                raise RuntimeError(
                    f"Scorer returned {len(scored)} scores for "
                    f"{len(missing_idx)} texts -- length mismatch."
                )
            for i, s in zip(missing_idx, scored):
                cached_by_key[keys[i]] = {
                    "pos": float(s["pos"]),
                    "neg": float(s["neg"]),
                    "neu": float(s["neu"]),
                    "score": float(s["score"]),
                }

        # Materialize the score columns in input order.
        score_frame["pos"] = [cached_by_key[k]["pos"] for k in keys]
        score_frame["neg"] = [cached_by_key[k]["neg"] for k in keys]
        score_frame["neu"] = [cached_by_key[k]["neu"] for k in keys]
        score_frame["score"] = [cached_by_key[k]["score"] for k in keys]

        # Write the union back to the cache ONLY if there were misses. When
        # every key was a cache hit, the existing cache is already correct
        # (idempotency: zero forward passes => zero writes). This keeps the
        # warm path fast (no parquet rewrite), satisfying the S2.2 DoD that
        # warm-run latency is < 5% of cold-run latency.
        if missing_idx:
            out_rows: list[dict[str, Any]] = []
            # Existing cached rows not in this batch (preserve them).
            input_keys = set(keys)
            if not cached.empty:
                for row in cached.itertuples(index=False):
                    if row.article_key not in input_keys:
                        out_rows.append(
                            {
                                "article_key": row.article_key,
                                "model_name": row.model_name,
                                "model_revision": row.model_revision,
                                "pos": float(row.pos),
                                "neg": float(row.neg),
                                "neu": float(row.neu),
                                "score": float(row.score),
                            }
                        )
            # Current input rows (newly scored or cache-hit).
            for row in score_frame.itertuples(index=False):
                out_rows.append(
                    {
                        "article_key": row.article_key,
                        "model_name": row.model_name,
                        "model_revision": row.model_revision,
                        "pos": float(row.pos),
                        "neg": float(row.neg),
                        "neu": float(row.neu),
                        "score": float(row.score),
                    }
                )
            union = pd.DataFrame(out_rows, columns=CANONICAL_PER_ARTICLE_COLUMNS)
            self._write_cache(union, scorer)

        # Build the returned frame: score cols + passthrough input cols
        # (all input cols except text_col). Score cols override on collision.
        passthrough = articles.drop(columns=[text_col]).reset_index(drop=True)
        overlap = [c for c in passthrough.columns if c in CANONICAL_PER_ARTICLE_COLUMNS]
        if overlap:
            passthrough = passthrough.drop(columns=overlap)
        out = pd.concat([score_frame.reset_index(drop=True), passthrough], axis=1)
        return out

    def get_meta(self) -> dict[str, Any] | None:
        """Return the parsed ``_meta.json`` (or ``None`` if missing)."""
        if not self.meta_path.exists():
            return None
        try:
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("Could not parse %s (%s).", self.meta_path, exc)
            return None


__all__ = [
    "SentimentCache",
    "compute_article_key",
    "DEFAULT_CACHE_DIR",
    "CACHE_FILE_NAME",
    "META_FILE_NAME",
]
