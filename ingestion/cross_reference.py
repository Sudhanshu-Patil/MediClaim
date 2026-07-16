"""Static cross-reference resolution at ingestion time (README §9).

Detects "see section 4.2" / "refer to clause 3" / "as defined in Appendix B"
style patterns in chunk text and resolves them to the PARENT chunk of the
referenced section within the same document. Resolved references become
``(:Chunk)-[:REFERENCES]->(:Chunk)`` edges in Neo4j — the common case is
pre-linked for free, with zero query-time latency.

References that cannot be resolved statically (cross-document targets, or
sections that don't exist in this doc) are returned as unresolved; tasks.py
stores them on the chunk payload so the later agentic MCP tool
(``resolve_reference``) can handle the long tail at query time.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from pydantic import BaseModel

from ingestion.metadata_schema import ChunkType, IngestChunk

logger = logging.getLogger(__name__)

# "section 4.2", "clause 3", "appendix B", "annexure 2", "table 5", "schedule C"
_TARGET = r"(?P<kind>section|sub-?section|clause|appendix|annex(?:ure)?|table|schedule|part)"
_LABEL = r"(?P<label>\d+(?:\.\d+)*|[A-Z]\b)"

REFERENCE_RE = re.compile(
    r"(?:see(?:\s+also)?|refer(?:\s+to)?|as\s+(?:described|defined|outlined|specified|detailed|listed)\s+in|"
    r"under|per|pursuant\s+to|in\s+accordance\s+with|described\s+in|listed\s+in|according\s+to)"
    r"\s+" + _TARGET + r"\s+" + _LABEL,
    re.IGNORECASE,
)

# Section number at the start of a heading: "4.2 Exclusions", "3. Coverage",
# or "Appendix B — Forms".
_HEADING_NUM_RE = re.compile(r"^\s*(?:section\s+)?(\d+(?:\.\d+)*)[.)\s]", re.IGNORECASE)
_HEADING_APPENDIX_RE = re.compile(
    r"^\s*(appendix|annex(?:ure)?|schedule|part)\s+([A-Z]\b|\d+)", re.IGNORECASE
)


class ReferenceEdge(BaseModel):
    source_chunk_id: str
    target_chunk_id: Optional[str] = None
    kind: str            # section / clause / appendix / ...
    target_label: str    # "4.2", "B", ...
    raw_text: str        # the matched phrase, for auditability
    resolved: bool = False


def _normalize_kind(kind: str) -> str:
    kind = kind.lower().replace("-", "")
    if kind in {"subsection"}:
        return "section"
    if kind in {"annex", "annexure"}:
        return "appendix"
    return kind


def _section_key(kind: str, label: str) -> str:
    # Sections and clauses share a numbering namespace in most policy docs.
    kind = _normalize_kind(kind)
    if kind in {"section", "clause"}:
        kind = "section"
    return f"{kind}:{label.lower().rstrip('.')}"


def build_section_index(chunks: list[IngestChunk]) -> dict[str, str]:
    """Map 'section:4.2' / 'appendix:b' → parent chunk_id for this document."""
    index: dict[str, str] = {}
    for chunk in chunks:
        if chunk.metadata.chunk_type is not ChunkType.PARENT:
            continue
        title = chunk.metadata.section_title or ""
        match = _HEADING_NUM_RE.match(title)
        if match:
            key = _section_key("section", match.group(1))
            index.setdefault(key, chunk.chunk_id)
            continue
        match = _HEADING_APPENDIX_RE.match(title)
        if match:
            key = _section_key(match.group(1), match.group(2))
            index.setdefault(key, chunk.chunk_id)
    return index


def detect_references(chunks: list[IngestChunk]) -> list[ReferenceEdge]:
    """Scan child + table chunks for reference phrases; resolve within-doc."""
    section_index = build_section_index(chunks)
    edges: list[ReferenceEdge] = []
    seen: set[tuple[str, str, str]] = set()

    for chunk in chunks:
        # Parents aggregate their children's text — scanning them too would
        # only duplicate every edge at a coarser anchor.
        if chunk.metadata.chunk_type not in (ChunkType.CHILD, ChunkType.TABLE):
            continue
        for match in REFERENCE_RE.finditer(chunk.text):
            kind = _normalize_kind(match.group("kind"))
            label = match.group("label").rstrip(".")
            dedupe_key = (chunk.chunk_id, kind, label.lower())
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            target_id = section_index.get(_section_key(kind, label))
            # A chunk inside section 4.2 saying "see section 4.2" is noise.
            if target_id == chunk.metadata.parent_chunk_id:
                continue
            edges.append(
                ReferenceEdge(
                    source_chunk_id=chunk.chunk_id,
                    target_chunk_id=target_id,
                    kind=kind,
                    target_label=label,
                    raw_text=match.group(0),
                    resolved=target_id is not None,
                )
            )

    resolved = sum(1 for e in edges if e.resolved)
    logger.info(
        "Cross-references: %d detected, %d resolved statically, %d left for "
        "query-time resolve_reference",
        len(edges),
        resolved,
        len(edges) - resolved,
    )
    return edges
