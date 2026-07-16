"""Neo4j graph store client (README §3, §9).

Graph model:

    (:Document {uid, doc_id, version, content_hash, status, ...})
        -[:HAS_CHUNK]->  (:Chunk {chunk_id, ...metadata schema...})
    (:Document new)  -[:SUPERSEDES]->  (:Document old)      # audit history
    (:Chunk parent)  -[:HAS_CHILD]->   (:Chunk child)       # hierarchy
    (:Chunk summary) -[:SUMMARIZES]->  (:Chunk table)       # atomic tables
    (:Chunk a)       -[:REFERENCES]->  (:Chunk b)           # "see section X.X"

Document nodes exist for EVERY ingested document — Neo4j doubles as the
version registry that freshness.py consults. Chunk nodes are only created for
graph-scoped source types (policy, clinical_guideline) per README §3; claim
notes stay vector-only.

Community Edition note: node-key (composite) constraints are Enterprise-only,
so Document uniqueness uses a single ``uid`` property = ``"{doc_id}:v{version}"``.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from neo4j import GraphDatabase

from config import get_settings

logger = logging.getLogger(__name__)

_CONSTRAINTS = [
    "CREATE CONSTRAINT document_uid IF NOT EXISTS FOR (d:Document) REQUIRE d.uid IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE INDEX document_doc_id IF NOT EXISTS FOR (d:Document) ON (d.doc_id)",
    "CREATE INDEX chunk_doc_id IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
    "CREATE INDEX chunk_status IF NOT EXISTS FOR (c:Chunk) ON (c.status)",
]


def document_uid(doc_id: str, version: int) -> str:
    return f"{doc_id}:v{version}"


class Neo4jStore:
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.driver = GraphDatabase.driver(
            uri or settings.neo4j_uri,
            auth=(user or settings.neo4j_user, password or settings.neo4j_password),
        )

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> "Neo4jStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ── Schema ──────────────────────────────────────────────────────────────
    def ensure_constraints(self) -> None:
        with self.driver.session() as session:
            for statement in _CONSTRAINTS:
                session.run(statement)

    # ── Document version registry (freshness.py) ───────────────────────────
    def get_latest_document(self, doc_id: str, status: Optional[str] = None) -> Optional[dict]:
        query = (
            "MATCH (d:Document {doc_id: $doc_id}) "
            "WHERE $status IS NULL OR d.status = $status "
            "RETURN d ORDER BY d.version DESC LIMIT 1"
        )
        with self.driver.session() as session:
            record = session.run(query, doc_id=doc_id, status=status).single()
            return dict(record["d"]) if record else None

    def upsert_document(self, props: dict) -> None:
        """MERGE a Document node by uid; idempotent on retries."""
        query = "MERGE (d:Document {uid: $uid}) SET d += $props"
        with self.driver.session() as session:
            session.run(query, uid=props["uid"], props=props)

    def mark_document_superseded(self, uid: str, superseded_by: str) -> None:
        query = (
            "MATCH (d:Document {uid: $uid}) "
            "SET d.status = 'superseded', d.superseded_by = $superseded_by"
        )
        with self.driver.session() as session:
            session.run(query, uid=uid, superseded_by=superseded_by)

    def create_supersedes_edge(self, new_uid: str, old_uid: str) -> None:
        """(new)-[:SUPERSEDES]->(old): audit trail of version history (README §3)."""
        query = (
            "MATCH (new:Document {uid: $new_uid}), (old:Document {uid: $old_uid}) "
            "MERGE (new)-[:SUPERSEDES]->(old)"
        )
        with self.driver.session() as session:
            session.run(query, new_uid=new_uid, old_uid=old_uid)

    # ── Chunks (graph-scoped source types only) ─────────────────────────────
    def upsert_chunks(self, doc_uid: str, chunk_props: Sequence[dict]) -> int:
        """MERGE chunk nodes + HAS_CHUNK / HAS_CHILD / SUMMARIZES edges."""
        if not chunk_props:
            return 0
        upsert_query = """
        MATCH (d:Document {uid: $doc_uid})
        UNWIND $rows AS row
        MERGE (c:Chunk {chunk_id: row.chunk_id})
        SET c += row
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN count(c) AS n
        """
        parent_query = """
        UNWIND $rows AS row
        WITH row WHERE row.parent_chunk_id IS NOT NULL
        MATCH (p:Chunk {chunk_id: row.parent_chunk_id})
        MATCH (c:Chunk {chunk_id: row.chunk_id})
        MERGE (p)-[:HAS_CHILD]->(c)
        """
        summary_query = """
        UNWIND $rows AS row
        WITH row WHERE row.refers_to_chunk_id IS NOT NULL
        MATCH (s:Chunk {chunk_id: row.chunk_id})
        MATCH (t:Chunk {chunk_id: row.refers_to_chunk_id})
        MERGE (s)-[:SUMMARIZES]->(t)
        """
        rows = list(chunk_props)
        with self.driver.session() as session:
            n = session.run(upsert_query, doc_uid=doc_uid, rows=rows).single()["n"]
            session.run(parent_query, rows=rows)
            session.run(summary_query, rows=rows)
        logger.info("Upserted %d chunk nodes for %s", n, doc_uid)
        return n

    def mark_chunks_superseded(
        self, doc_id: str, doc_version: int, superseded_by: str
    ) -> int:
        query = (
            "MATCH (c:Chunk {doc_id: $doc_id, doc_version: $doc_version}) "
            "SET c.status = 'superseded', c.superseded_by = $superseded_by "
            "RETURN count(c) AS n"
        )
        with self.driver.session() as session:
            result = session.run(
                query, doc_id=doc_id, doc_version=doc_version, superseded_by=superseded_by
            )
            return result.single()["n"]

    # ── Cross-references (README §9 static path) ────────────────────────────
    def create_reference_edges(self, edges: Sequence[dict]) -> int:
        """(:Chunk)-[:REFERENCES {kind, target_label, raw_text}]->(:Chunk).

        Each edge dict: source_chunk_id, target_chunk_id, kind, target_label,
        raw_text. MERGE keys on the endpoints + target_label so re-ingestion
        never duplicates edges.
        """
        if not edges:
            return 0
        query = """
        UNWIND $edges AS edge
        MATCH (s:Chunk {chunk_id: edge.source_chunk_id})
        MATCH (t:Chunk {chunk_id: edge.target_chunk_id})
        MERGE (s)-[r:REFERENCES {target_label: edge.target_label}]->(t)
        SET r.kind = edge.kind, r.raw_text = edge.raw_text, r.resolved = true
        RETURN count(r) AS n
        """
        with self.driver.session() as session:
            result = session.run(query, edges=list(edges))
            n = result.single()["n"]
        logger.info("Created/updated %d REFERENCES edges", n)
        return n
