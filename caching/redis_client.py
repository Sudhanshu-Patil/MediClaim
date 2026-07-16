"""Redis client + L3 embedding cache (README §6).

Redis plays four roles in the full system; Phase 1 implements:
  * L3 embedding cache — ``chunk_hash → embedding vector``. On re-ingestion,
    unchanged chunks (same content hash) skip re-embedding entirely. At
    100k-doc scale this is the difference between minutes and hours.
  * Celery broker/result backend (wired in ingestion/tasks.py, separate DBs).

L1 (exact-match answer cache) and L2 (semantic cache) are query-time concerns
and arrive with the LangGraph agent in a later phase; the key layout here
leaves them room (``l1:``/``l2:`` prefixes).

Vectors are stored as raw little-endian float32 bytes (~3 KB for a 768-dim
bge-base vector) — no JSON overhead.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

import numpy as np
import redis

from config import get_settings

logger = logging.getLogger(__name__)

_pools: dict[str, redis.ConnectionPool] = {}


def get_redis(url: Optional[str] = None) -> redis.Redis:
    """Pooled Redis client. decode_responses stays False — we store raw bytes."""
    url = url or get_settings().redis_url
    if url not in _pools:
        _pools[url] = redis.ConnectionPool.from_url(url, decode_responses=False)
    return redis.Redis(connection_pool=_pools[url])


class EmbeddingCache:
    """L3 cache: chunk_hash → float32 vector.

    Keys are namespaced by embedding model so switching models never serves
    stale vectors: ``l3:emb:{model}:{chunk_hash}``.
    """

    def __init__(
        self,
        client: Optional[redis.Redis] = None,
        model_name: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.client = client or get_redis()
        self.model_name = model_name or settings.embedding_model
        # 0 / None → no expiry: embeddings of immutable content never go stale.
        ttl = settings.l3_embedding_cache_ttl if ttl_seconds is None else ttl_seconds
        self.ttl_seconds = ttl if ttl and ttl > 0 else None

    def _key(self, chunk_hash: str) -> str:
        return f"l3:emb:{self.model_name}:{chunk_hash}"

    def get_many(self, chunk_hashes: Sequence[str]) -> dict[str, list[float]]:
        """Return {chunk_hash: vector} for every cache hit."""
        if not chunk_hashes:
            return {}
        raw = self.client.mget([self._key(h) for h in chunk_hashes])
        hits: dict[str, list[float]] = {}
        for chunk_hash, blob in zip(chunk_hashes, raw):
            if blob is not None:
                hits[chunk_hash] = np.frombuffer(blob, dtype=np.float32).tolist()
        return hits

    def set_many(self, vectors: dict[str, Iterable[float]]) -> None:
        if not vectors:
            return
        pipe = self.client.pipeline(transaction=False)
        for chunk_hash, vector in vectors.items():
            blob = np.asarray(list(vector), dtype=np.float32).tobytes()
            if self.ttl_seconds:
                pipe.set(self._key(chunk_hash), blob, ex=self.ttl_seconds)
            else:
                pipe.set(self._key(chunk_hash), blob)
        pipe.execute()

    def stats(self) -> dict[str, int]:
        """Approximate entry count for this model's namespace (debug helper)."""
        count = 0
        for _ in self.client.scan_iter(match=f"l3:emb:{self.model_name}:*", count=1000):
            count += 1
        return {"entries": count}
