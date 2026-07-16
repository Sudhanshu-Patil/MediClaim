"""Identity-based freshness & versioning (README §3, §8).

Freshness is NOT timestamp-guessing:
  * ``doc_id``       — stable identity of the *logical* document, derived from
                       its normalized name + source_type (or an explicit
                       ``logical_name`` override). "Outpatient Policy" keeps
                       the same doc_id across every uploaded revision.
  * ``content_hash`` — sha256 of the file bytes; detects real change.
  * ``version``      — monotonically increasing int per doc_id, assigned from
                       the Neo4j Document registry.

Decision on upload:
  * no Document node for doc_id            → NEW
  * latest node has same content_hash      → UNCHANGED (skip re-ingestion)
  * latest node has different content_hash → CHANGED (ingest new version,
      then mark the old version's chunks ``status: superseded`` +
      ``superseded_by`` in BOTH stores, and create
      (new)-[:SUPERSEDES]->(old) in Neo4j)

Old chunks are never deleted: claims can be disputed, so past answers must
remain explainable. They simply stop matching the default
``status = active`` retrieval filter.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from ingestion.metadata_schema import SourceType
from retrieval.graph_store import Neo4jStore, document_uid
from retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)

# Strip version-ish suffixes so "policy_v2.pdf" and "policy (3).pdf" map to
# the same logical document as "policy.pdf".
_VERSION_SUFFIX_RE = re.compile(
    r"[\s_\-.]*(?:v(?:er(?:sion)?)?[\s_\-.]*\d+(?:\.\d+)*|\(\d+\)|final|draft|copy)$",
    re.IGNORECASE,
)


class VersionAction(str, Enum):
    NEW = "new"
    CHANGED = "changed"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class VersionDecision:
    action: VersionAction
    doc_id: str
    content_hash: str
    new_version: int
    previous_version: Optional[int] = None
    previous_uid: Optional[str] = None

    @property
    def new_uid(self) -> str:
        return document_uid(self.doc_id, self.new_version)


def normalize_logical_name(file_name: str) -> str:
    stem = Path(file_name).stem.strip().lower()
    stem = re.sub(r"[\s_\-]+", " ", stem)  # "outpatient_policy" == "outpatient policy"
    return _VERSION_SUFFIX_RE.sub("", stem).strip() or stem


def compute_doc_id(
    file_path: str | Path,
    source_type: SourceType,
    logical_name: Optional[str] = None,
) -> str:
    """Stable logical-document identity: hash of source_type + normalized name."""
    name = logical_name.strip().lower() if logical_name else normalize_logical_name(
        Path(file_path).name
    )
    return hashlib.sha256(f"{source_type.value}:{name}".encode("utf-8")).hexdigest()[:24]


def compute_content_hash(file_path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class FreshnessManager:
    """Consults the Neo4j Document registry and applies supersede transitions."""

    def __init__(self, graph_store: Neo4jStore, vector_store: QdrantStore) -> None:
        self.graph = graph_store
        self.vectors = vector_store

    def check(
        self,
        file_path: str | Path,
        source_type: SourceType,
        logical_name: Optional[str] = None,
    ) -> VersionDecision:
        doc_id = compute_doc_id(file_path, source_type, logical_name)
        content_hash = compute_content_hash(file_path)
        latest = self.graph.get_latest_document(doc_id)
        latest_active = self.graph.get_latest_document(doc_id, status="active")

        if latest is None:
            decision = VersionDecision(
                action=VersionAction.NEW,
                doc_id=doc_id,
                content_hash=content_hash,
                new_version=1,
            )
        elif (
            latest.get("status") == "active"
            and latest.get("content_hash") == content_hash
        ):
            decision = VersionDecision(
                action=VersionAction.UNCHANGED,
                doc_id=doc_id,
                content_hash=content_hash,
                new_version=int(latest["version"]),
                previous_version=int(latest["version"]),
                previous_uid=latest["uid"],
            )
        else:
            # Changed content — or a leftover status:"ingesting" marker from a
            # crashed run. A partial version with the SAME hash is redone under
            # the same version number: deterministic chunk IDs make the redo a
            # pure overwrite, never a duplicate.
            if (
                latest.get("status") != "active"
                and latest.get("content_hash") == content_hash
            ):
                new_version = int(latest["version"])
            else:
                new_version = int(latest["version"]) + 1
            decision = VersionDecision(
                action=VersionAction.CHANGED if latest_active else VersionAction.NEW,
                doc_id=doc_id,
                content_hash=content_hash,
                new_version=new_version,
                previous_version=int(latest_active["version"]) if latest_active else None,
                previous_uid=latest_active["uid"] if latest_active else None,
            )

        logger.info(
            "Freshness check %s: %s (doc_id=%s, version=%d)",
            Path(file_path).name,
            decision.action.value,
            doc_id,
            decision.new_version,
        )
        return decision

    def register_version(
        self,
        decision: VersionDecision,
        *,
        file_name: str,
        source_type: SourceType,
        effective_date: Optional[str] = None,
        ingestion_timestamp: Optional[str] = None,
        num_chunks: int = 0,
        status: str = "active",
    ) -> None:
        """Create/refresh the Document node for the newly ingested version.

        Called twice per ingestion: with status="ingesting" before the chunk
        writes (so HAS_CHUNK edges have a node to attach to) and with
        status="active" as the final commit marker once every store write has
        succeeded. Freshness only trusts active markers.
        """
        self.graph.upsert_document(
            {
                "uid": decision.new_uid,
                "doc_id": decision.doc_id,
                "version": decision.new_version,
                "content_hash": decision.content_hash,
                "file_name": file_name,
                "source_type": source_type.value,
                "status": status,
                "effective_date": effective_date,
                "ingestion_timestamp": ingestion_timestamp,
                "num_chunks": num_chunks,
            }
        )

    def supersede_previous(self, decision: VersionDecision) -> None:
        """Apply the CHANGED transition after the new version is fully ingested.

        Ordering matters: the new version's chunks are already active in both
        stores, so there is never a window where a query sees no active
        version of the document.
        """
        if decision.action is not VersionAction.CHANGED:
            return
        assert decision.previous_version is not None and decision.previous_uid is not None

        self.vectors.mark_superseded(
            decision.doc_id, decision.previous_version, superseded_by=decision.new_uid
        )
        n = self.graph.mark_chunks_superseded(
            decision.doc_id, decision.previous_version, superseded_by=decision.new_uid
        )
        self.graph.mark_document_superseded(
            decision.previous_uid, superseded_by=decision.new_uid
        )
        self.graph.create_supersedes_edge(decision.new_uid, decision.previous_uid)
        logger.info(
            "Superseded %s (marked %d graph chunks): %s now active",
            decision.previous_uid,
            n,
            decision.new_uid,
        )
