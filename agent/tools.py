"""Shared tool implementations (README §9, tech-stack MCP row).

One implementation, two consumers:
  * agent/mcp_server.py exposes these as MCP tools for any MCP client
    (Claude Desktop, other agents).
  * agent/nodes/resolve.py calls resolve_reference deterministically inside
    the graph (the capped query-time fallback of README §9).

The claims "database" is a synthetic SQLite seeded by scripts/seed_claims.py —
a stand-in for the adjudication system of record.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Optional

CLAIMS_DB = Path(__file__).resolve().parents[1] / "data" / "claims.db"


# ── resolve_reference (README §9 dynamic path) ──────────────────────────────
def resolve_reference(section_number: str, doc_id: Optional[str] = None) -> Optional[dict]:
    """Look up the parent chunk of a section like '3.2' or 'Appendix A'.

    Returns {chunk_id, doc_id, section_title, text, page_number} or None.
    """
    from retrieval.graph_store import Neo4jStore

    label = section_number.strip().rstrip(".")
    graph = Neo4jStore()
    try:
        query = """
        MATCH (c:Chunk {chunk_type: 'parent', status: 'active'})
        WHERE ($doc_id IS NULL OR c.doc_id = $doc_id)
          AND (c.section_title STARTS WITH $label
               OR c.section_title STARTS WITH ('Section ' + $label)
               OR toLower(c.section_title) STARTS WITH toLower($label))
        RETURN c.chunk_id AS chunk_id, c.doc_id AS doc_id,
               c.section_title AS section_title, c.text AS text,
               c.page_number AS page_number
        LIMIT 1
        """
        with graph.driver.session() as session:
            record = session.run(query, label=label, doc_id=doc_id).single()
            return dict(record) if record else None
    finally:
        graph.close()


# ── policy_lookup ───────────────────────────────────────────────────────────
def policy_lookup(term: str, top_k: int = 3) -> list[dict]:
    """Retrieve the most relevant active policy chunks for a term/question."""
    from retrieval.retriever import HybridRetriever

    retriever = HybridRetriever()
    try:
        results = retriever.retrieve(term, top_k=top_k)
        return [
            {"chunk_id": r.chunk_id, "doc_name": r.doc_name,
             "doc_version": r.doc_version, "section_title": r.section_title,
             "page_number": r.page_number, "text": r.text[:1500],
             "score": r.score}
            for r in results
        ]
    finally:
        retriever.close()


# ── claim_db_query / fraud_flag (synthetic claims store) ───────────────────
def _conn() -> sqlite3.Connection:
    if not CLAIMS_DB.exists():
        raise FileNotFoundError(
            f"{CLAIMS_DB} missing — run: python scripts/seed_claims.py"
        )
    conn = sqlite3.connect(CLAIMS_DB)
    conn.row_factory = sqlite3.Row
    return conn


def claim_db_query(
    claim_id: Optional[str] = None,
    member_id: Optional[str] = None,
    procedure_code: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """Query the claims store by any combination of filters."""
    clauses, params = [], []
    for column, value in [("claim_id", claim_id), ("member_id", member_id),
                          ("procedure_code", procedure_code), ("status", status)]:
        if value:
            clauses.append(f"{column} = ?")
            params.append(value)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with _conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM claims {where} ORDER BY service_date DESC LIMIT ?",
            (*params, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def fraud_flag(claim_id: str, reason: str) -> dict:
    """Flag a claim for fraud review (writes an audit row, sets status)."""
    if not re.fullmatch(r"CLM-\d{5}", claim_id):
        return {"ok": False, "error": f"invalid claim_id format: {claim_id}"}
    with _conn() as conn:
        row = conn.execute("SELECT claim_id FROM claims WHERE claim_id = ?",
                           (claim_id,)).fetchone()
        if not row:
            return {"ok": False, "error": f"claim {claim_id} not found"}
        conn.execute(
            "INSERT INTO fraud_flags (claim_id, reason) VALUES (?, ?)",
            (claim_id, reason),
        )
        conn.execute(
            "UPDATE claims SET status = 'fraud_review' WHERE claim_id = ?",
            (claim_id,),
        )
    return {"ok": True, "claim_id": claim_id, "status": "fraud_review",
            "reason": reason}
