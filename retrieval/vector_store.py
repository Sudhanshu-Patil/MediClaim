"""Qdrant vector store client (README §4, §7).

* One collection, cosine distance, bge-base 768-dim vectors.
* Payload indexes on ``status``, ``doc_id``, ``source_type``, ``doc_version``
  and ``chunk_type`` so query-time filters (``status = active``, per-doc
  scoping) stay fast at 100k-doc scale.
* Superseding a document version is a payload update on its points — old
  chunks are kept (claims can be disputed; audit history must survive), they
  just stop matching the default ``status = active`` filter.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from config import get_settings

logger = logging.getLogger(__name__)

_PAYLOAD_INDEXES: dict[str, qm.PayloadSchemaType] = {
    "status": qm.PayloadSchemaType.KEYWORD,
    "doc_id": qm.PayloadSchemaType.KEYWORD,
    "source_type": qm.PayloadSchemaType.KEYWORD,
    "chunk_type": qm.PayloadSchemaType.KEYWORD,
    "doc_version": qm.PayloadSchemaType.INTEGER,
}


class QdrantStore:
    def __init__(
        self,
        url: Optional[str] = None,
        collection: Optional[str] = None,
        vector_size: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.collection = collection or settings.qdrant_collection
        self.vector_size = vector_size or settings.embedding_dim
        self.client = QdrantClient(url=url or settings.qdrant_url)

    # ── Schema ──────────────────────────────────────────────────────────────
    def ensure_collection(self) -> None:
        """Create the collection + payload indexes if missing. Idempotent."""
        if not self.client.collection_exists(self.collection):
            logger.info("Creating Qdrant collection %s", self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=qm.VectorParams(
                    size=self.vector_size,
                    distance=qm.Distance.COSINE,
                ),
                # Scalar quantization: README §7 — millions of vectors on one
                # local instance.
                quantization_config=qm.ScalarQuantization(
                    scalar=qm.ScalarQuantizationConfig(
                        type=qm.ScalarType.INT8, always_ram=True
                    )
                ),
            )
        for field, schema in _PAYLOAD_INDEXES.items():
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception:
                pass  # index already exists

    # ── Writes ──────────────────────────────────────────────────────────────
    def upsert_chunks(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict],
        batch_size: int = 128,
    ) -> int:
        """Batched upsert. Deterministic IDs make re-runs overwrite, not duplicate."""
        total = 0
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self.client.upsert(
                collection_name=self.collection,
                points=qm.Batch(
                    ids=list(ids[start:end]),
                    vectors=[list(v) for v in vectors[start:end]],
                    payloads=list(payloads[start:end]),
                ),
                wait=True,
            )
            total += len(ids[start:end])
        logger.info("Upserted %d points into %s", total, self.collection)
        return total

    def mark_superseded(self, doc_id: str, doc_version: int, superseded_by: str) -> None:
        """Flip every chunk of (doc_id, doc_version) to status=superseded."""
        self.client.set_payload(
            collection_name=self.collection,
            payload={"status": "superseded", "superseded_by": superseded_by},
            points=qm.Filter(
                must=[
                    qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)),
                    qm.FieldCondition(
                        key="doc_version", match=qm.MatchValue(value=doc_version)
                    ),
                ]
            ),
            wait=True,
        )
        logger.info(
            "Marked chunks of %s v%d superseded by %s", doc_id, doc_version, superseded_by
        )

    # ── Reads (verification / later phases) ────────────────────────────────
    def count(self, doc_id: Optional[str] = None, status: Optional[str] = None) -> int:
        must = []
        if doc_id:
            must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)))
        if status:
            must.append(qm.FieldCondition(key="status", match=qm.MatchValue(value=status)))
        result = self.client.count(
            collection_name=self.collection,
            count_filter=qm.Filter(must=must) if must else None,
            exact=True,
        )
        return result.count
