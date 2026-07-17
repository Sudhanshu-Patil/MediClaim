"""Batched, L3-cached embedding for ingestion (README §6 + §7).

* Model: BAAI/bge-base-en-v1.5 via fastembed (ONNX, CPU, local, zero cost).
* Batching: 64–128 chunks per call (README §7) — configurable via
  EMBED_BATCH_SIZE.
* L3 cache: chunk_hash → vector in Redis; unchanged chunks on re-ingestion
  are never re-embedded.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from config import get_settings
from caching.redis_client import EmbeddingCache

logger = logging.getLogger(__name__)


class CachedEmbedder:
    def __init__(
        self,
        cache: Optional[EmbeddingCache] = None,
        model_name: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.batch_size = batch_size or settings.embed_batch_size
        self.cache = cache if cache is not None else EmbeddingCache(model_name=self.model_name)
        self._model = None
        self.last_cache_hits = 0
        self.last_cache_misses = 0

    def _get_model(self):
        # Lazy: loading the ONNX model takes seconds; skip it entirely when
        # every chunk is an L3 cache hit.
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info("Loading embedding model %s", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    def embed(self, texts: Sequence[str], chunk_hashes: Sequence[str]) -> list[list[float]]:
        """Embed texts, consulting the L3 cache by chunk_hash first.

        ``texts`` and ``chunk_hashes`` are parallel sequences; the returned
        vectors are in the same order.
        """
        if len(texts) != len(chunk_hashes):
            raise ValueError("texts and chunk_hashes must be parallel sequences")

        cached = self.cache.get_many(chunk_hashes) if self.cache else {}
        self.last_cache_hits = len(cached)

        miss_indices = [i for i, h in enumerate(chunk_hashes) if h not in cached]
        self.last_cache_misses = len(miss_indices)

        vectors: list[Optional[list[float]]] = [
            cached.get(h) for h in chunk_hashes
        ]

        if miss_indices:
            model = self._get_model()
            newly_embedded: dict[str, list[float]] = {}
            for start in range(0, len(miss_indices), self.batch_size):
                batch_idx = miss_indices[start : start + self.batch_size]
                batch_texts = [texts[i] for i in batch_idx]
                batch_vectors = [v.tolist() for v in model.embed(batch_texts)]
                for i, vec in zip(batch_idx, batch_vectors):
                    vectors[i] = vec
                    newly_embedded[chunk_hashes[i]] = vec
            if self.cache:
                self.cache.set_many(newly_embedded)

        logger.info(
            "Embedded %d chunks: %d L3 cache hits, %d computed",
            len(texts),
            self.last_cache_hits,
            self.last_cache_misses,
        )
        return [v for v in vectors if v is not None]

    # ── Sparse BM25 (hybrid retrieval's lexical arm; no cache needed —
    #    tokenization-only, no neural model, microseconds per text) ─────────
    _sparse_model = None

    def _get_sparse(self):
        if CachedEmbedder._sparse_model is None:
            from fastembed import SparseTextEmbedding

            logger.info("Loading sparse BM25 model Qdrant/bm25")
            CachedEmbedder._sparse_model = SparseTextEmbedding("Qdrant/bm25")
        return CachedEmbedder._sparse_model

    def embed_sparse(self, texts: Sequence[str]) -> list[tuple[list[int], list[float]]]:
        return [(e.indices.tolist(), e.values.tolist())
                for e in self._get_sparse().embed(list(texts))]

    def embed_query_sparse(self, query: str) -> tuple[list[int], list[float]]:
        e = next(iter(self._get_sparse().query_embed(query)))
        return e.indices.tolist(), e.values.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Query-side embedding (retrieval).

        Uses fastembed's query_embed so models with an asymmetric query
        instruction (bge's "Represent this sentence for searching...") get it
        applied — passages and queries must not be embedded identically.
        Cached under a "q::" hash so repeated queries skip the model.
        """
        import hashlib

        query_hash = "q::" + hashlib.sha256(query.encode("utf-8")).hexdigest()
        if self.cache:
            hit = self.cache.get_many([query_hash])
            if query_hash in hit:
                return hit[query_hash]

        model = self._get_model()
        try:
            vector = next(iter(model.query_embed(query))).tolist()
        except AttributeError:  # older fastembed without query_embed
            vector = next(iter(model.embed([query]))).tolist()

        if self.cache:
            self.cache.set_many({query_hash: vector})
        return vector
