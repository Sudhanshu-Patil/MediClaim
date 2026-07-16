"""Docling wrapper: PDF / DOCX / PPTX → normalized ParsedDocument.

Responsibilities (README §5):
  * TableFormer (ACCURATE mode) reconstructs real row/column structure for
    complex fixed-column, multi-row tables instead of flattening them.
  * Multi-page tables are merged into ONE logical ParsedTable. Docling already
    merges many continuation tables itself (a TableItem then carries multiple
    provenance entries); ``_merge_split_tables`` additionally merges adjacent
    same-width tables that Docling emitted as separate items across a page
    break.
  * Provenance per element: page_number + bbox for PDF, paragraph_index for
    DOCX, slide_index for PPTX — this is what makes citations human-checkable
    and line-level highlighting possible (README §8).

Parsing happens once, at ingestion time, never at query time.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, TableItem, TextItem

from ingestion.metadata_schema import BoundingBox

logger = logging.getLogger(__name__)

_SUFFIX_TO_FORMAT = {
    ".pdf": InputFormat.PDF,
    ".docx": InputFormat.DOCX,
    ".pptx": InputFormat.PPTX,
}

# Boilerplate labels that pollute chunks and citations.
_SKIP_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
_HEADING_LABELS = {DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE}


class SourceFormat(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"


class ParsedBlock(BaseModel):
    kind: Literal["heading", "paragraph", "list_item"]
    text: str
    heading_level: Optional[int] = None
    page_number: Optional[int] = None
    slide_index: Optional[int] = None
    paragraph_index: Optional[int] = None
    bboxes: list[BoundingBox] = Field(default_factory=list)


class ParsedTable(BaseModel):
    kind: Literal["table"] = "table"
    markdown: str
    num_rows: int
    num_cols: int
    caption: Optional[str] = None
    page_numbers: list[int] = Field(default_factory=list)
    slide_index: Optional[int] = None
    paragraph_index: Optional[int] = None
    bboxes: list[BoundingBox] = Field(default_factory=list)


ParsedElement = Union[ParsedBlock, ParsedTable]


class ParsedDocument(BaseModel):
    source_path: str
    file_name: str
    source_format: SourceFormat
    elements: list[ParsedElement]

    @property
    def tables(self) -> list[ParsedTable]:
        return [e for e in self.elements if isinstance(e, ParsedTable)]


def _build_converter(use_pypdfium_backend: bool = False) -> DocumentConverter:
    # Keep peak memory low: process one page at a time (throughput at scale
    # comes from Celery parallelism across documents, not page batching).
    try:
        from docling.datamodel.settings import settings as docling_settings

        docling_settings.perf.page_batch_size = int(
            os.getenv("DOCLING_PAGE_BATCH_SIZE", "1")
        )
    except Exception:
        pass

    pdf_options = PdfPipelineOptions()
    # OCR models add several hundred MB of RAM and only matter for scanned
    # documents; digital-born PDFs parse fine without. Set DOCLING_DO_OCR=1
    # when ingesting scans.
    pdf_options.do_ocr = os.getenv("DOCLING_DO_OCR", "0") == "1"
    pdf_options.do_table_structure = True
    # ACCURATE TableFormer: required for merged cells and complex
    # fixed-column, multi-row tables (README §5).
    pdf_options.table_structure_options.mode = TableFormerMode.ACCURATE
    pdf_options.table_structure_options.do_cell_matching = True

    if use_pypdfium_backend:
        # Fallback backend for the known std::bad_alloc crash in the default
        # docling-parse C++ backend (docling issue #3671, frequent on Windows).
        # With pdfium's coarser text cells, TableFormer predicts its own cells.
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

        pdf_options.table_structure_options.do_cell_matching = False
        pdf_format_option = PdfFormatOption(
            pipeline_options=pdf_options, backend=PyPdfiumDocumentBackend
        )
    else:
        pdf_format_option = PdfFormatOption(pipeline_options=pdf_options)

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.PPTX],
        format_options={InputFormat.PDF: pdf_format_option},
    )


_converters: dict[bool, DocumentConverter] = {}


def get_converter(use_pypdfium_backend: bool = False) -> DocumentConverter:
    # Module-level singletons: Docling loads model weights (TableFormer,
    # layout) on construction — reuse across Celery task invocations.
    if use_pypdfium_backend not in _converters:
        _converters[use_pypdfium_backend] = _build_converter(use_pypdfium_backend)
    return _converters[use_pypdfium_backend]


def _convert_with_fallback(path: Path, is_pdf: bool):
    """Convert with the default backend; retry PDFs on FULL or PARTIAL failure.

    Docling reports pages it could not process as PARTIAL_SUCCESS instead of
    raising. Accepting that silently would ingest a document with missing
    pages — a claims answer could then cite an incomplete policy. So partial
    PDF parses retry on pypdfium2, and a still-partial result is an error
    unless DOCLING_ALLOW_PARTIAL=1 explicitly opts into best-effort.
    """
    from docling.datamodel.base_models import ConversionStatus
    from docling.exceptions import ConversionError

    def _errors(result) -> list[str]:
        return [str(getattr(e, "error_message", e)) for e in (result.errors or [])]

    try:
        result = get_converter().convert(path)
        if result.status == ConversionStatus.SUCCESS:
            return result
        if not is_pdf:
            logger.warning("Partial conversion of %s accepted (no alternate backend): %s",
                           path.name, _errors(result))
            return result
        logger.warning(
            "docling-parse converted %s only partially (%s); retrying with pypdfium2",
            path.name, _errors(result),
        )
    except ConversionError:
        if not is_pdf:
            raise
        logger.warning(
            "docling-parse backend failed on %s; retrying with pypdfium2 backend",
            path.name,
        )

    result = get_converter(use_pypdfium_backend=True).convert(path)
    if result.status != ConversionStatus.SUCCESS:
        if os.getenv("DOCLING_ALLOW_PARTIAL", "0") == "1":
            logger.error("Accepting PARTIAL conversion of %s (DOCLING_ALLOW_PARTIAL=1): %s",
                         path.name, _errors(result))
        else:
            raise RuntimeError(
                f"Docling only partially converted {path.name} "
                f"(failures: {_errors(result)}). Refusing to ingest an incomplete "
                "document; set DOCLING_ALLOW_PARTIAL=1 to accept best-effort parses."
            )
    return result


def _provenance(item) -> tuple[list[int], list[BoundingBox]]:
    """Extract (page numbers, bounding boxes) from a Docling item's prov list."""
    pages: list[int] = []
    bboxes: list[BoundingBox] = []
    for prov in getattr(item, "prov", None) or []:
        page_no = getattr(prov, "page_no", None)
        if page_no is None:
            continue
        if page_no not in pages:
            pages.append(page_no)
        bb = getattr(prov, "bbox", None)
        if bb is not None:
            origin = getattr(bb, "coord_origin", None)
            bboxes.append(
                BoundingBox(
                    page_no=page_no,
                    l=bb.l,
                    t=bb.t,
                    r=bb.r,
                    b=bb.b,
                    coord_origin=str(getattr(origin, "value", origin) or "BOTTOMLEFT"),
                )
            )
    return pages, bboxes


def _table_to_markdown(item: TableItem, doc) -> str:
    try:
        return item.export_to_markdown(doc=doc)
    except TypeError:  # older docling-core signature
        return item.export_to_markdown()


def _table_caption(item: TableItem, doc) -> Optional[str]:
    try:
        caption = item.caption_text(doc)
        return caption or None
    except Exception:
        return None


def _table_dims(item: TableItem) -> tuple[int, int]:
    data = getattr(item, "data", None)
    return (getattr(data, "num_rows", 0) or 0, getattr(data, "num_cols", 0) or 0)


def _merge_split_tables(elements: list[ParsedElement]) -> list[ParsedElement]:
    """Merge consecutive tables Docling emitted separately across a page break.

    Heuristic: two tables are one logical table when they are adjacent in
    reading order (nothing but the page break between them), have the same
    column count, and the second one starts on the page right after the first
    one ends. The continuation's repeated header row, if identical, is dropped.
    """
    merged: list[ParsedElement] = []
    for element in elements:
        prev = merged[-1] if merged else None
        if (
            isinstance(element, ParsedTable)
            and isinstance(prev, ParsedTable)
            and element.num_cols == prev.num_cols
            and element.num_cols > 0
            and prev.page_numbers
            and element.page_numbers
            and element.page_numbers[0] - prev.page_numbers[-1] == 1
        ):
            prev_lines = prev.markdown.strip().splitlines()
            cont_lines = element.markdown.strip().splitlines()
            dropped_header = 0
            # Drop repeated "| header |" + "|---|" rows on the continuation.
            while (
                cont_lines
                and dropped_header < 2
                and dropped_header < len(prev_lines)
                and cont_lines[0].strip() == prev_lines[dropped_header].strip()
            ):
                cont_lines.pop(0)
                dropped_header += 1
            prev.markdown = "\n".join(prev_lines + cont_lines)
            prev.num_rows += element.num_rows - (1 if dropped_header else 0)
            prev.page_numbers.extend(
                p for p in element.page_numbers if p not in prev.page_numbers
            )
            prev.bboxes.extend(element.bboxes)
            if element.caption and not prev.caption:
                prev.caption = element.caption
            logger.info(
                "Merged continuation table on page %s into table starting page %s",
                element.page_numbers[0],
                prev.page_numbers[0],
            )
            continue
        merged.append(element)
    return merged


def parse_document(file_path: str | Path) -> ParsedDocument:
    """Parse a PDF/DOCX/PPTX into a normalized, provenance-rich structure."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    input_format = _SUFFIX_TO_FORMAT.get(suffix)
    if input_format is None:
        raise ValueError(
            f"Unsupported file type {suffix!r} — expected one of {sorted(_SUFFIX_TO_FORMAT)}"
        )
    source_format = SourceFormat(suffix.lstrip("."))

    logger.info("Parsing %s with Docling (%s)", path.name, source_format.value)
    result = _convert_with_fallback(path, is_pdf=source_format is SourceFormat.PDF)
    doc = result.document

    elements: list[ParsedElement] = []
    paragraph_counter = 0  # DOCX: running index over body items

    for item, _level in doc.iterate_items():
        label = getattr(item, "label", None)
        if label in _SKIP_LABELS:
            continue

        if isinstance(item, TableItem):
            pages, bboxes = _provenance(item)
            num_rows, num_cols = _table_dims(item)
            markdown = _table_to_markdown(item, doc)
            if not markdown.strip():
                continue
            elements.append(
                ParsedTable(
                    markdown=markdown,
                    num_rows=num_rows,
                    num_cols=num_cols,
                    caption=_table_caption(item, doc),
                    page_numbers=pages,
                    slide_index=(pages[0] if pages and source_format is SourceFormat.PPTX else None),
                    paragraph_index=(paragraph_counter if source_format is SourceFormat.DOCX else None),
                    bboxes=bboxes if source_format is SourceFormat.PDF else [],
                )
            )
            paragraph_counter += 1
            continue

        if isinstance(item, TextItem):
            text = (getattr(item, "text", "") or "").strip()
            if not text:
                continue
            pages, bboxes = _provenance(item)

            if label in _HEADING_LABELS:
                kind = "heading"
                heading_level = getattr(item, "level", 1) or 1
            elif label == DocItemLabel.LIST_ITEM:
                kind = "list_item"
                heading_level = None
            else:
                kind = "paragraph"
                heading_level = None

            elements.append(
                ParsedBlock(
                    kind=kind,
                    text=text,
                    heading_level=heading_level,
                    page_number=(pages[0] if pages and source_format is SourceFormat.PDF else None),
                    slide_index=(pages[0] if pages and source_format is SourceFormat.PPTX else None),
                    paragraph_index=(paragraph_counter if source_format is SourceFormat.DOCX else None),
                    bboxes=bboxes if source_format is SourceFormat.PDF else [],
                )
            )
            paragraph_counter += 1

    elements = _merge_split_tables(elements)

    logger.info(
        "Parsed %s: %d elements (%d tables)",
        path.name,
        len(elements),
        sum(1 for e in elements if isinstance(e, ParsedTable)),
    )
    return ParsedDocument(
        source_path=str(path.resolve()),
        file_name=path.name,
        source_format=source_format,
        elements=elements,
    )
