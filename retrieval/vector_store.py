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
        api_key: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.collection = collection or settings.qdrant_collection
        self.vector_size = vector_size or settings.embedding_dim
        # api_key is None for local Docker Qdrant (no auth); Qdrant Cloud
        # requires it — QdrantClient accepts api_key=None harmlessly.
        self.client = QdrantClient(
            url=url or settings.qdrant_url,
            api_key=api_key or settings.qdrant_api_key,
        )

    # ── Schema ──────────────────────────────────────────────────────────────
    def ensure_collection(self) -> None:
        """Create the collection + payload indexes if missing. Idempotent.

        Schema: named dense vector ("dense", bge cosine) + named sparse
        vector ("bm25", server-side IDF) — true dense+sparse hybrid across
        ALL source types (the Neo4j Lucene arm only covers graph-scoped docs).
        """
        if not self.client.collection_exists(self.collection):
            logger.info("Creating Qdrant collection %s (dense+bm25)", self.collection)
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": qm.VectorParams(
                        size=self.vector_size, distance=qm.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)
                },
                # Scalar quantization: README §7 — millions of vectors on one
                # local instance.
                quantization_config=qm.ScalarQuantization(
                    scalar=qm.ScalarQuantizationConfig(
                        type=qm.ScalarType.INT8, always_ram=True
                    )
                ),
            )
        else:
            info = self.client.get_collection(self.collection)
            if not (info.config.params.sparse_vectors or {}).get("bm25"):
                logger.critical(
                    "Collection %s predates the sparse-BM25 schema. Sparse "
                    "retrieval is DISABLED until you recreate + re-ingest: "
                    "curl -X DELETE .../collections/%s && re-run ingestion.",
                    self.collection, self.collection,
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
        sparse_vectors: Optional[Sequence[tuple[list[int], list[float]]]] = None,
        batch_size: int = 128,
    ) -> int:
        """Batched upsert. Deterministic IDs make re-runs overwrite, not duplicate."""
        total = 0
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            points = []
            for offset, point_id in enumerate(ids[start:end]):
                i = start + offset
                vector: dict = {"dense": list(vectors[i])}
                if sparse_vectors is not None:
                    indices, values = sparse_vectors[i]
                    vector["bm25"] = qm.SparseVector(indices=indices, values=values)
                points.append(qm.PointStruct(id=point_id, vector=vector,
                                             payload=payloads[i]))
            self.client.upsert(collection_name=self.collection, points=points,
                               wait=True)
            total += len(points)
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

    # ── Query-time reads ────────────────────────────────────────────────────
    def search(
        self,
        query_vector: Sequence[float],
        top_n: int = 20,
        status: str = "active",
        chunk_types: Optional[Sequence[str]] = None,
        source_type: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> list[dict]:
        """Vector search, excluding superseded chunks by default (README §3).

        Returns [{chunk_id, score, payload}, ...] best-first.
        """
        must: list[qm.Condition] = [
            qm.FieldCondition(key="status", match=qm.MatchValue(value=status))
        ]
        if chunk_types:
            must.append(
                qm.FieldCondition(key="chunk_type", match=qm.MatchAny(any=list(chunk_types)))
            )
        if source_type:
            must.append(
                qm.FieldCondition(key="source_type", match=qm.MatchValue(value=source_type))
            )
        if doc_id:
            must.append(qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id)))

        result = self.client.query_points(
            collection_name=self.collection,
            query=list(query_vector),
            using="dense",
            query_filter=qm.Filter(must=must),
            limit=top_n,
            with_payload=True,
        )
        return [
            {"chunk_id": str(p.id), "score": p.score, "payload": p.payload}
            for p in result.points
        ]

    def search_sparse(
        self,
        indices: list[int],
        values: list[float],
        top_n: int = 20,
        status: str = "active",
        chunk_types: Optional[Sequence[str]] = None,
        source_type: Optional[str] = None,
    ) -> list[dict]:
        """BM25 sparse search — the lexical arm of hybrid retrieval (exact
        codes, names, serial-style tokens that embeddings blur)."""
        must: list[qm.Condition] = [
            qm.FieldCondition(key="status", match=qm.MatchValue(value=status))
        ]
        if chunk_types:
            must.append(qm.FieldCondition(key="chunk_type",
                                          match=qm.MatchAny(any=list(chunk_types))))
        if source_type:
            must.append(qm.FieldCondition(key="source_type",
                                          match=qm.MatchValue(value=source_type)))
        try:
            result = self.client.query_points(
                collection_name=self.collection,
                query=qm.SparseVector(indices=indices, values=values),
                using="bm25",
                query_filter=qm.Filter(must=must),
                limit=top_n,
                with_payload=True,
            )
        except Exception:
            logger.warning("sparse search unavailable (old collection schema?)",
                           exc_info=True)
            return []
        return [
            {"chunk_id": str(p.id), "score": p.score, "payload": p.payload}
            for p in result.points
        ]

    def retrieve(self, chunk_ids: Sequence[str]) -> dict[str, dict]:
        """Fetch payloads by ID (used for graph-only hits after fusion)."""
        if not chunk_ids:
            return {}
        points = self.client.retrieve(
            collection_name=self.collection,
            ids=list(chunk_ids),
            with_payload=True,
        )
        return {str(p.id): p.payload for p in points}

    # ── Document library (UI browsing) ──────────────────────────────────────
    def list_documents(
        self, source_type: Optional[str] = None, status: str = "active"
    ) -> list[dict]:
        """Group chunks by doc_id into a library-style listing for the UI.

        Returns one entry per document: doc_id, doc_name, doc_version,
        source_type, effective_date, ingestion_timestamp, num_chunks,
        num_tables, sections (sample of section titles).
        """
        must: list[qm.Condition] = [
            qm.FieldCondition(key="status", match=qm.MatchValue(value=status))
        ]
        if source_type:
            must.append(
                qm.FieldCondition(key="source_type", match=qm.MatchValue(value=source_type))
            )
        scroll_filter = qm.Filter(must=must)

        docs: dict[str, dict] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=scroll_filter,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                if not doc_id:
                    continue
                entry = docs.setdefault(doc_id, {
                    "doc_id": doc_id,
                    "doc_name": payload.get("doc_name"),
                    "doc_version": payload.get("doc_version"),
                    "source_type": payload.get("source_type"),
                    "effective_date": payload.get("effective_date"),
                    "ingestion_timestamp": payload.get("ingestion_timestamp"),
                    "num_chunks": 0,
                    "num_tables": 0,
                    "sections": set(),
                })
                entry["num_chunks"] += 1
                if payload.get("chunk_type") == "table":
                    entry["num_tables"] += 1
                if payload.get("section_title"):
                    entry["sections"].add(payload["section_title"])
            if offset is None:
                break

        results = []
        for entry in docs.values():
            entry["sections"] = sorted(entry["sections"])[:8]
            results.append(entry)
        results.sort(key=lambda d: (d.get("doc_name") or "").lower())
        return results

    def get_document_name(self, doc_id: str) -> Optional[str]:
        """One payload's doc_name for a doc_id — used to locate the source file."""
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=qm.Filter(
                must=[qm.FieldCondition(key="doc_id", match=qm.MatchValue(value=doc_id))]
            ),
            limit=1,
            with_payload=True,
        )
        return points[0].payload.get("doc_name") if points else None

    # ── Verification helpers ────────────────────────────────────────────────
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
