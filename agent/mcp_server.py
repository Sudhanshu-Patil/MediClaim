"""Custom MCP server (README §4 tech stack, §9 dynamic reference resolution).

Exposes the MedClaim tools over the Model Context Protocol so any MCP client
(Claude Desktop, another agent, an IDE) can adjudicate against the live
stores:

    resolve_reference — section-number → chunk (the long-tail fallback for
                        cross-references that weren't pre-linked at ingestion)
    policy_lookup     — hybrid retrieval over active policy chunks
    claim_db_query    — synthetic claims system-of-record (SQLite)
    fraud_flag        — flag a claim for SIU review (audited write)

Run over stdio (e.g. for Claude Desktop config):

    python agent/mcp_server.py

Requires the Docker stack up; claims tools need `python scripts/seed_claims.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp.server.fastmcp import FastMCP

from agent import tools

mcp = FastMCP(
    "medclaim",
    instructions=(
        "Tools for insurance-claims adjudication against the MedClaim "
        "policy corpus and claims store. Prefer policy_lookup for policy "
        "questions; resolve_reference for 'see section X.Y' targets; "
        "claim_db_query before flagging anything with fraud_flag."
    ),
)


@mcp.tool()
def resolve_reference(section_number: str, doc_id: Optional[str] = None) -> dict:
    """Resolve a policy section reference (e.g. '3.2' or 'Appendix A') to its
    full section text. Optionally scope to one doc_id."""
    result = tools.resolve_reference(section_number, doc_id)
    return result or {"error": f"section {section_number!r} not found in active documents"}


@mcp.tool()
def policy_lookup(term: str, top_k: int = 3) -> list[dict]:
    """Hybrid-search the active policy corpus (vector + graph + rerank) and
    return the top matching chunks with citation metadata."""
    return tools.policy_lookup(term, top_k=top_k)


@mcp.tool()
def claim_db_query(
    claim_id: Optional[str] = None,
    member_id: Optional[str] = None,
    procedure_code: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 10,
) -> list[dict]:
    """Query claims by any combination of claim_id, member_id,
    procedure_code, status (paid/denied/pending/fraud_review)."""
    return tools.claim_db_query(claim_id, member_id, procedure_code, status, limit)


@mcp.tool()
def fraud_flag(claim_id: str, reason: str) -> dict:
    """Flag a claim (CLM-#####) for fraud review with a stated reason.
    Writes an audit row and sets the claim status to fraud_review."""
    return tools.fraud_flag(claim_id, reason)


if __name__ == "__main__":
    mcp.run()
