"""Typed per-chunk metadata schema — implements README §5 exactly.

Required fields (README §5 table):
    chunk_id, doc_id, doc_version, section_title,
    page_number / slide_index / paragraph_index, bbox, source_type,
    effective_date, status, doc_hash, ingestion_timestamp

Plus pipeline-internal fields (chunk_type, parent_chunk_id, chunk_hash, ...)
that the parent-child hierarchy, atomic-table handling, and L3 embedding
cache need. Serializers are provided for both stores:

  * ``to_qdrant_payload()`` — JSON-safe dict; Qdrant payload indexes are
    created on ``status``, ``doc_id``, ``source_type``, ``doc_version`` and
    ``chunk_type`` (see retrieval/vector_store.py).
  * ``to_neo4j_props()`` — flat primitives only (Neo4j properties cannot hold
    nested maps, so ``bbox`` is stored as a JSON string).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# Stable namespace for deterministic chunk UUIDs: uuid5(NS, doc_id:version:index).
# Re-running ingestion on the same content always yields the same IDs, so
# Qdrant/Neo4j upserts are idempotent (README §7).
CHUNK_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "medclaim-agentic-rag/chunk")


class SourceType(str, Enum):
    """README §5: policy / clinical-guideline / claim-note."""

    POLICY = "policy"
    CLINICAL_GUIDELINE = "clinical_guideline"
    CLAIM_NOTE = "claim_note"


class ChunkStatus(str, Enum):
    """README §3: retrieval filters exclude superseded chunks by default."""

    ACTIVE = "active"
    SUPERSEDED = "superseded"


class ChunkType(str, Enum):
    PARENT = "parent"            # header + whole section (context expansion)
    CHILD = "child"              # sentence-window prose chunk
    TABLE = "table"              # atomic — never split
    TABLE_SUMMARY = "table_summary"  # 1–2 line summary pointing at the table


class BoundingBox(BaseModel):
    """One highlightable region on one PDF page (README §8 line-level highlighting).

    Coordinates follow Docling's convention (l, t, r, b) with an explicit
    coord_origin (usually BOTTOMLEFT for PDFs).
    """

    page_no: int
    l: float
    t: float
    r: float
    b: float
    coord_origin: str = "BOTTOMLEFT"


def make_chunk_id(doc_id: str, doc_version: int, chunk_index: int) -> str:
    """Deterministic chunk id (README §5): hash of doc_id + version + index.

    UUIDv5 string — valid as a Qdrant point ID and unique in Neo4j.
    """
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, f"{doc_id}:v{doc_version}:{chunk_index}"))


def make_chunk_hash(text: str) -> str:
    """Content hash of a chunk's text — key for the Redis L3 embedding cache."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ChunkMetadata(BaseModel):
    # ── README §5 required fields ──────────────────────────────────────────
    chunk_id: str
    doc_id: str
    doc_version: int
    section_title: Optional[str] = None
    page_number: Optional[int] = None      # PDF
    slide_index: Optional[int] = None      # PPTX
    paragraph_index: Optional[int] = None  # DOCX
    bbox: Optional[list[BoundingBox]] = None  # list: a chunk/table may span pages
    source_type: SourceType
    effective_date: Optional[date] = None
    status: ChunkStatus = ChunkStatus.ACTIVE
    doc_hash: str
    ingestion_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    # ── Versioning (README §3) ─────────────────────────────────────────────
    superseded_by: Optional[str] = None  # uid ("doc_id:vN") of the newer version

    # ── Pipeline fields ────────────────────────────────────────────────────
    chunk_type: ChunkType
    chunk_index: int
    parent_chunk_id: Optional[str] = None      # child/table → its section parent
    refers_to_chunk_id: Optional[str] = None   # table_summary → full table chunk
    chunk_hash: str                            # L3 cache key (README §6)
    doc_name: Optional[str] = None             # human-readable source file name

    def to_qdrant_payload(self) -> dict:
        payload = self.model_dump(mode="json", exclude_none=True)
        return payload

    def to_neo4j_props(self) -> dict:
        props = self.model_dump(mode="json", exclude_none=True)
        if self.bbox is not None:
            props["bbox"] = json.dumps(props["bbox"])
        return props


class IngestChunk(BaseModel):
    """A chunk ready for embedding + upsert: text plus its full metadata."""

    text: str
    metadata: ChunkMetadata

    @property
    def chunk_id(self) -> str:
        return self.metadata.chunk_id
