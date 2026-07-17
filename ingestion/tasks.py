"""Celery ingestion pipeline (README §7, architecture diagram UPLOAD subgraph).

Pipeline per document:
    freshness check → Docling parse → hierarchical chunk + metadata
    → cross-reference detection → embed (Redis L3-cached, batched)
    → Qdrant upsert → Neo4j upsert (graph-scoped types) → register version
    → supersede previous version (if CHANGED)

Reliability properties:
  * ``acks_late`` + ``task_reject_on_worker_lost`` — a job survives a worker
    crash mid-flight and is redelivered.
  * ``autoretry_for`` + exponential backoff with jitter, capped retries.
  * Idempotent end to end: chunk IDs are deterministic (uuid5 of
    doc_id:version:index), Qdrant upserts overwrite by ID, Neo4j writes are
    MERGE — re-running a job (or retrying half-way) never duplicates data.
  * The UNCHANGED short-circuit means re-queueing an already-ingested file is
    a cheap no-op.

Run a worker (from the repo root):
    celery -A ingestion.tasks worker --loglevel=info --concurrency=2
    (on Windows add: --pool=solo)
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

from celery import Celery

from config import get_settings
from ingestion.chunker import chunk_document
from ingestion.cross_reference import detect_references
from ingestion.embedder import CachedEmbedder
from ingestion.freshness import FreshnessManager, VersionAction
from ingestion.metadata_schema import SourceType
from ingestion.parser import parse_document
from retrieval.graph_store import Neo4jStore
from retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)

_settings = get_settings()

celery_app = Celery(
    "medclaim_ingestion",
    broker=_settings.celery_broker_url,
    backend=_settings.celery_result_backend,
)
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,   # long-running parse jobs: no hoarding
    task_track_started=True,
    result_expires=7 * 24 * 3600,
    broker_connection_retry_on_startup=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)


def run_ingestion(
    file_path: str,
    source_type: str = SourceType.POLICY.value,
    effective_date: Optional[str] = None,
    logical_name: Optional[str] = None,
) -> dict:
    """The full pipeline, callable directly (--sync) or from the Celery task."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {file_path}")
    src_type = SourceType(source_type)
    eff_date = date.fromisoformat(effective_date) if effective_date else None
    settings = get_settings()

    vector_store = QdrantStore()
    graph_store = Neo4jStore()
    try:
        # Idempotent schema setup (collection, payload indexes, constraints).
        vector_store.ensure_collection()
        graph_store.ensure_constraints()

        freshness = FreshnessManager(graph_store, vector_store)

        # 1 ── Version check (README §3): new / changed / unchanged
        decision = freshness.check(path, src_type, logical_name)
        if decision.action is VersionAction.UNCHANGED:
            return {
                "action": "skipped_unchanged",
                "file": path.name,
                "doc_id": decision.doc_id,
                "doc_version": decision.new_version,
            }

        # 2 ── Parse (Docling + TableFormer)
        parsed = parse_document(path)

        # 3 ── Chunk + attach metadata (README §5)
        chunks = chunk_document(
            parsed,
            doc_id=decision.doc_id,
            doc_version=decision.new_version,
            source_type=src_type,
            doc_hash=decision.content_hash,
            effective_date=eff_date,
        )
        if not chunks:
            raise ValueError(f"Docling produced no content for {path.name}")

        # 4 ── Cross-reference detection (README §9, static path)
        edges = detect_references(chunks)
        resolved_edges = [e for e in edges if e.resolved]
        unresolved_by_chunk: dict[str, list[dict]] = {}
        for e in edges:
            if not e.resolved:
                unresolved_by_chunk.setdefault(e.source_chunk_id, []).append(
                    {"kind": e.kind, "target_label": e.target_label, "raw_text": e.raw_text}
                )

        # 5 ── Embed, consulting the Redis L3 cache (README §6).
        #      Dense (bge) + sparse BM25 (lexical arm for hybrid retrieval —
        #      tokenization-only, no cache needed).
        embedder = CachedEmbedder()
        vectors = embedder.embed(
            [c.text for c in chunks], [c.metadata.chunk_hash for c in chunks]
        )
        sparse_vectors = embedder.embed_sparse([c.text for c in chunks])

        # 6 ── Qdrant upsert (deterministic IDs → idempotent)
        payloads = []
        for chunk in chunks:
            payload = chunk.metadata.to_qdrant_payload()
            payload["text"] = chunk.text
            refs = unresolved_by_chunk.get(chunk.chunk_id)
            if refs:
                # Long-tail input for the query-time resolve_reference MCP tool.
                payload["unresolved_references"] = refs
            payloads.append(payload)
        vector_store.upsert_chunks(
            [c.chunk_id for c in chunks], vectors, payloads,
            sparse_vectors=sparse_vectors,
            batch_size=settings.embed_batch_size,
        )

        # 7 ── Neo4j: chunk nodes + edges (only for graph-scoped source types,
        #      README §3), then the Document registry node LAST — it is the
        #      commit marker the UNCHANGED short-circuit trusts, so it must
        #      only exist once every store write above has succeeded.
        graph_scoped = src_type.value in settings.graph_scoped_source_types
        n_edges = 0
        # The chunk upsert needs the Document node to attach HAS_CHUNK edges;
        # create it as non-active first, flip to active as the final commit.
        freshness.register_version(
            decision,
            file_name=path.name,
            source_type=src_type,
            effective_date=effective_date,
            ingestion_timestamp=datetime.now(timezone.utc).isoformat(),
            num_chunks=len(chunks),
            status="ingesting",
        )
        if graph_scoped:
            chunk_props = []
            for chunk in chunks:
                props = chunk.metadata.to_neo4j_props()
                props["text"] = chunk.text
                chunk_props.append(props)
            graph_store.upsert_chunks(decision.new_uid, chunk_props)
            n_edges = graph_store.create_reference_edges(
                [e.model_dump(exclude={"resolved"}) for e in resolved_edges]
            )
        # 8 ── Supersede the previous version. The new chunks are already
        #      queryable, so there is never a window with no active version;
        #      doing this before the final commit marker means a crash here is
        #      healed by the retry instead of leaving two active versions.
        freshness.supersede_previous(decision)

        # 9 ── Final commit marker: only now does the freshness check treat
        #      this version as fully ingested.
        freshness.register_version(
            decision,
            file_name=path.name,
            source_type=src_type,
            effective_date=effective_date,
            ingestion_timestamp=datetime.now(timezone.utc).isoformat(),
            num_chunks=len(chunks),
            status="active",
        )

        return {
            "action": decision.action.value,
            "file": path.name,
            "doc_id": decision.doc_id,
            "doc_version": decision.new_version,
            "doc_uid": decision.new_uid,
            "superseded": decision.previous_uid,
            "chunks": len(chunks),
            "tables": sum(1 for c in chunks if c.metadata.chunk_type.value == "table"),
            "reference_edges": n_edges,
            "unresolved_references": sum(len(v) for v in unresolved_by_chunk.values()),
            "embedding_cache_hits": embedder.last_cache_hits,
            "embedding_cache_misses": embedder.last_cache_misses,
            "graph_scoped": graph_scoped,
        }
    finally:
        graph_store.close()


@celery_app.task(
    bind=True,
    name="ingestion.ingest_document",
    acks_late=True,
    autoretry_for=(Exception,),
    retry_backoff=5,          # 5s, 10s, 20s, 40s, ... exponential
    retry_backoff_max=300,
    retry_jitter=True,
    max_retries=5,
)
def ingest_document(
    self,
    file_path: str,
    source_type: str = SourceType.POLICY.value,
    effective_date: Optional[str] = None,
    logical_name: Optional[str] = None,
) -> dict:
    """Async ingestion of one document. Safe to retry / re-queue at any point."""
    result = run_ingestion(file_path, source_type, effective_date, logical_name)
    logger.info("Ingestion result: %s", result)
    return result


@celery_app.task(name="ingestion.ingest_directory")
def ingest_directory(
    directory: str,
    source_type: str = SourceType.POLICY.value,
    effective_date: Optional[str] = None,
) -> dict:
    """Fan out one ingest_document job per supported file in a directory."""
    root = Path(directory)
    queued = []
    for path in sorted(root.iterdir()):
        if path.suffix.lower() in {".pdf", ".docx", ".pptx"}:
            ingest_document.delay(str(path), source_type, effective_date)
            queued.append(path.name)
    return {"queued": queued, "count": len(queued)}
