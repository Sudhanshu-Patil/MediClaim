"""Deterministic cross-reference resolution node (README §9 dynamic path).

Between retrieve and generate: if the *query* or the retrieved chunks point
at a section ("see section 3.2") that is NOT already among the retrieved
context, fetch that section's parent chunk directly from Neo4j and append it.

Deliberately deterministic rather than LLM tool-calling: the fine-tuned 3B
is format-locked to cited-JSON generation and unreliable at free-form tool
use, and README §9 caps this path at 1–2 calls anyway — a regex + graph
lookup implements the same contract with zero extra latency variance. The
same implementation is exposed to real agentic clients via agent/mcp_server.py.
"""

from __future__ import annotations

import logging
import re

from agent.state import AgentState

logger = logging.getLogger(__name__)

MAX_RESOLUTIONS = 2  # README §9: capped to bound latency

_SECTION_REF_RE = re.compile(
    r"\b(?:section|clause|appendix)\s+(\d+(?:\.\d+)*|[A-Z])\b", re.IGNORECASE
)


def resolve_references(state: AgentState) -> dict:
    from agent.tools import resolve_reference

    chunks = list(state.get("chunks", []))
    if not chunks:
        return {}
    have_ids = {c["chunk_id"] for c in chunks}
    have_sections = " ".join(f"§{c.get('section_title') or ''}" for c in chunks).lower()

    # Candidate labels from the query + the retrieved texts themselves.
    text_pool = state.get("query", "") + " " + " ".join(c["text"][:2000] for c in chunks[:4])
    labels: list[str] = []
    for match in _SECTION_REF_RE.finditer(text_pool):
        label = match.group(1).rstrip(".")
        if label not in labels:
            labels.append(label)

    added = 0
    for label in labels:
        if added >= MAX_RESOLUTIONS:
            break
        if label.lower() in have_sections:  # already retrieved
            continue
        result = resolve_reference(label)
        if not result or result["chunk_id"] in have_ids:
            continue
        chunks.append({
            "chunk_id": result["chunk_id"],
            "text": result["text"],
            "score": 0.0,
            "fused_score": 0.0,
            "reranked": False,
            "chunk_type": "parent",
            "doc_id": result["doc_id"],
            "doc_version": 0,
            "doc_name": None,
            "section_title": result["section_title"],
            "page_number": result.get("page_number"),
            "sources": ["resolve_reference"],
        })
        have_ids.add(result["chunk_id"])
        added += 1
        logger.info("resolve_reference: appended section %r (%s)",
                    result["section_title"], result["chunk_id"])

    return {"chunks": chunks} if added else {}
