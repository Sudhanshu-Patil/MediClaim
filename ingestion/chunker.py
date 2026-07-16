"""Hierarchical (parent-child) + sentence-window chunking. README §5.

Strategy per content type:
  * Prose      → sentence-window packing, 512–1024 tokens, ~10–20% overlap.
  * Tables     → ATOMIC: the whole table is one chunk, never split, plus an
                 auto-generated 1–2 line summary chunk whose
                 ``refers_to_chunk_id`` points back at the full table.
  * Sections   → parent-child: heading + full section text is the parent;
                 sentence-window chunks are its children and inherit its
                 metadata (section_title, provenance, versioning fields).

Chunk IDs are deterministic — uuid5(doc_id, version, running index) — so
re-running ingestion on identical content upserts the exact same points
instead of duplicating (README §7 idempotency).

Implementation note: this is a self-contained implementation of the
LlamaIndex-style HierarchicalNodeParser / sentence-window strategy named in
README §4. Writing it directly (~200 lines) keeps the dependency tree small
and makes chunk IDs, metadata inheritance, and table atomicity fully
deterministic; the behavior matches the README-specified strategy exactly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Callable, Optional

from config import get_settings
from ingestion.metadata_schema import (
    BoundingBox,
    ChunkMetadata,
    ChunkStatus,
    ChunkType,
    IngestChunk,
    SourceType,
    make_chunk_hash,
    make_chunk_id,
)
from ingestion.parser import ParsedBlock, ParsedDocument, ParsedTable

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])|\n{2,}")


# ── Token counting ──────────────────────────────────────────────────────────
def _build_token_counter() -> Callable[[str], int]:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(encoding.encode(text))
    except Exception:  # offline / tiktoken unavailable — ~1.3 tokens per word
        logger.warning("tiktoken unavailable; using word-count token estimate")
        return lambda text: max(1, int(len(text.split()) * 1.3))


_count_tokens: Optional[Callable[[str], int]] = None


def count_tokens(text: str) -> int:
    global _count_tokens
    if _count_tokens is None:
        _count_tokens = _build_token_counter()
    return _count_tokens(text)


# ── Sectioning ──────────────────────────────────────────────────────────────
@dataclass
class _Section:
    title: str
    heading_block: Optional[ParsedBlock]
    blocks: list[ParsedBlock]
    tables: list[ParsedTable]


def _split_into_sections(parsed: ParsedDocument) -> list[_Section]:
    """Group elements under their nearest preceding heading (reading order)."""
    sections: list[_Section] = []
    current = _Section(title=parsed.file_name, heading_block=None, blocks=[], tables=[])

    for element in parsed.elements:
        if isinstance(element, ParsedBlock) and element.kind == "heading":
            if current.blocks or current.tables or current.heading_block:
                sections.append(current)
            current = _Section(
                title=element.text.strip(), heading_block=element, blocks=[], tables=[]
            )
        elif isinstance(element, ParsedTable):
            current.tables.append(element)
        else:
            current.blocks.append(element)

    sections.append(current)
    return [s for s in sections if s.blocks or s.tables or s.heading_block]


# ── Sentence-window packing ─────────────────────────────────────────────────
@dataclass
class _Sentence:
    text: str
    tokens: int
    block: ParsedBlock  # provenance source


def _split_sentences(block: ParsedBlock) -> list[_Sentence]:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(block.text) if p and p.strip()]
    return [_Sentence(text=p, tokens=count_tokens(p), block=block) for p in parts]


def _hard_split(sentence: _Sentence, max_tokens: int) -> list[_Sentence]:
    """Split a pathological sentence longer than max_tokens on word boundaries."""
    words = sentence.text.split()
    pieces: list[_Sentence] = []
    step = max(1, int(len(words) * max_tokens / max(sentence.tokens, 1)))
    for i in range(0, len(words), step):
        piece = " ".join(words[i : i + step])
        pieces.append(_Sentence(text=piece, tokens=count_tokens(piece), block=sentence.block))
    return pieces


def _pack_windows(
    sentences: list[_Sentence], target: int, max_tokens: int, overlap: int
) -> list[list[_Sentence]]:
    """Pack sentences into windows of ~target tokens with sentence-level overlap."""
    normalized: list[_Sentence] = []
    for s in sentences:
        normalized.extend(_hard_split(s, max_tokens) if s.tokens > max_tokens else [s])

    windows: list[list[_Sentence]] = []
    window: list[_Sentence] = []
    window_tokens = 0
    carried = 0  # sentences at the window start that are overlap carry-over

    for sentence in normalized:
        if window and window_tokens + sentence.tokens > target:
            windows.append(window)
            # Carry trailing sentences forward as the 10–20% overlap.
            tail: list[_Sentence] = []
            tail_tokens = 0
            for prev in reversed(window):
                if tail_tokens + prev.tokens > overlap:
                    break
                tail.insert(0, prev)
                tail_tokens += prev.tokens
            window = list(tail)
            window_tokens = tail_tokens
            carried = len(tail)
        window.append(sentence)
        window_tokens += sentence.tokens

    # Emit the final window unless it is nothing but overlap carry-over.
    if window and len(window) > carried:
        windows.append(window)
    return windows


# ── Provenance helpers ──────────────────────────────────────────────────────
def _first_not_none(values):
    return next((v for v in values if v is not None), None)


def _blocks_provenance(
    blocks: list[ParsedBlock],
) -> tuple[Optional[int], Optional[int], Optional[int], Optional[list[BoundingBox]]]:
    pages = [b.page_number for b in blocks]
    slides = [b.slide_index for b in blocks]
    paragraphs = [b.paragraph_index for b in blocks]
    bboxes: list[BoundingBox] = []
    for b in blocks:
        bboxes.extend(b.bboxes)
    return (
        _first_not_none(pages),
        _first_not_none(slides),
        _first_not_none(paragraphs),
        bboxes or None,
    )


# ── Table summary (deterministic, zero-cost — no LLM at ingestion time) ────
def _summarize_table(table: ParsedTable, section_title: str, table_chunk_id: str) -> str:
    lines = [ln for ln in table.markdown.strip().splitlines() if ln.strip()]
    header_cells = []
    if lines:
        header_cells = [c.strip() for c in lines[0].strip("|").split("|") if c.strip()]
    columns = ", ".join(header_cells[:8]) or "unlabeled columns"
    label = table.caption or f"Table in section '{section_title}'"
    summary = (
        f"{label}: {table.num_rows} rows x {table.num_cols} columns "
        f"(columns: {columns})."
    )
    if table.page_numbers:
        pages = ", ".join(str(p) for p in table.page_numbers)
        summary += f" Spans page(s) {pages}."
    summary += f" Full table in chunk {table_chunk_id}."
    return summary


# ── Main entry point ────────────────────────────────────────────────────────
def chunk_document(
    parsed: ParsedDocument,
    *,
    doc_id: str,
    doc_version: int,
    source_type: SourceType,
    doc_hash: str,
    effective_date: Optional[date] = None,
) -> list[IngestChunk]:
    """Turn a ParsedDocument into parent/child/table/table-summary chunks."""
    settings = get_settings()
    ingestion_ts = datetime.now(timezone.utc)
    chunks: list[IngestChunk] = []
    chunk_index = 0

    def emit(
        text: str,
        chunk_type: ChunkType,
        *,
        section_title: Optional[str],
        page_number: Optional[int] = None,
        slide_index: Optional[int] = None,
        paragraph_index: Optional[int] = None,
        bbox: Optional[list[BoundingBox]] = None,
        parent_chunk_id: Optional[str] = None,
        refers_to_chunk_id: Optional[str] = None,
    ) -> IngestChunk:
        nonlocal chunk_index
        chunk_id = make_chunk_id(doc_id, doc_version, chunk_index)
        metadata = ChunkMetadata(
            chunk_id=chunk_id,
            doc_id=doc_id,
            doc_version=doc_version,
            section_title=section_title,
            page_number=page_number,
            slide_index=slide_index,
            paragraph_index=paragraph_index,
            bbox=bbox,
            source_type=source_type,
            effective_date=effective_date,
            status=ChunkStatus.ACTIVE,
            doc_hash=doc_hash,
            ingestion_timestamp=ingestion_ts,
            chunk_type=chunk_type,
            chunk_index=chunk_index,
            parent_chunk_id=parent_chunk_id,
            refers_to_chunk_id=refers_to_chunk_id,
            chunk_hash=make_chunk_hash(text),
            doc_name=parsed.file_name,
        )
        chunk_index += 1
        chunk = IngestChunk(text=text, metadata=metadata)
        chunks.append(chunk)
        return chunk

    for section in _split_into_sections(parsed):
        prose = "\n\n".join(b.text for b in section.blocks)
        parent_text = f"{section.title}\n\n{prose}".strip() if prose else section.title

        # Cap the parent (context-expansion payload, not a retrieval unit).
        if count_tokens(parent_text) > settings.parent_max_tokens:
            words = parent_text.split()
            keep = int(len(words) * settings.parent_max_tokens / count_tokens(parent_text))
            parent_text = " ".join(words[:keep]) + " …"

        prov_blocks = (
            [section.heading_block] if section.heading_block else []
        ) + section.blocks
        page, slide, para, bbox = _blocks_provenance(prov_blocks)

        parent = emit(
            parent_text,
            ChunkType.PARENT,
            section_title=section.title,
            page_number=page,
            slide_index=slide,
            paragraph_index=para,
            bbox=bbox,
        )

        # Children: sentence-window over the section's prose.
        sentences: list[_Sentence] = []
        for block in section.blocks:
            sentences.extend(_split_sentences(block))

        if sentences:
            total_tokens = sum(s.tokens for s in sentences)
            if total_tokens <= settings.chunk_max_tokens:
                windows = [sentences]
            else:
                windows = _pack_windows(
                    sentences,
                    target=settings.chunk_target_tokens,
                    max_tokens=settings.chunk_max_tokens,
                    overlap=settings.chunk_overlap_tokens,
                )
            for window in windows:
                window_blocks: list[ParsedBlock] = []
                seen_block_ids: set[int] = set()
                for s in window:
                    if id(s.block) not in seen_block_ids:
                        seen_block_ids.add(id(s.block))
                        window_blocks.append(s.block)
                w_page, w_slide, w_para, w_bbox = _blocks_provenance(window_blocks)
                emit(
                    " ".join(s.text for s in window),
                    ChunkType.CHILD,
                    section_title=section.title,
                    page_number=w_page,
                    slide_index=w_slide,
                    paragraph_index=w_para,
                    bbox=w_bbox,
                    parent_chunk_id=parent.chunk_id,
                )

        # Tables: atomic chunk + pointing summary chunk.
        for table in section.tables:
            table_text = (
                f"{table.caption}\n\n{table.markdown}" if table.caption else table.markdown
            )
            table_chunk = emit(
                table_text,
                ChunkType.TABLE,
                section_title=section.title,
                page_number=table.page_numbers[0] if table.page_numbers else None,
                slide_index=table.slide_index,
                paragraph_index=table.paragraph_index,
                bbox=table.bboxes or None,
                parent_chunk_id=parent.chunk_id,
            )
            emit(
                _summarize_table(table, section.title, table_chunk.chunk_id),
                ChunkType.TABLE_SUMMARY,
                section_title=section.title,
                page_number=table.page_numbers[0] if table.page_numbers else None,
                slide_index=table.slide_index,
                paragraph_index=table.paragraph_index,
                parent_chunk_id=parent.chunk_id,
                refers_to_chunk_id=table_chunk.chunk_id,
            )

    logger.info(
        "Chunked %s: %d chunks (%d parents, %d children, %d tables, %d table summaries)",
        parsed.file_name,
        len(chunks),
        sum(1 for c in chunks if c.metadata.chunk_type is ChunkType.PARENT),
        sum(1 for c in chunks if c.metadata.chunk_type is ChunkType.CHILD),
        sum(1 for c in chunks if c.metadata.chunk_type is ChunkType.TABLE),
        sum(1 for c in chunks if c.metadata.chunk_type is ChunkType.TABLE_SUMMARY),
    )
    return chunks
