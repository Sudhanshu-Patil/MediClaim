"""Shared state flowing through the LangGraph claim-query pipeline."""

from __future__ import annotations

from typing import Optional, TypedDict


class RetrievedChunkDict(TypedDict, total=False):
    chunk_id: str
    text: str
    score: float
    chunk_type: str
    doc_id: str
    doc_version: int
    doc_name: Optional[str]
    section_title: Optional[str]
    page_number: Optional[int]
    slide_index: Optional[int]
    paragraph_index: Optional[int]
    bbox: Optional[list]
    sources: list[str]


class AgentState(TypedDict, total=False):
    # input
    query: str
    source_type: Optional[str]

    # input guardrails (README §10)
    input_blocked: bool
    input_block_reason: Optional[str]
    pii_redacted: list[str]         # kinds of PII redacted from the query

    # router
    route: str                      # "vector" | "graph" | "hybrid"

    # retrieval
    chunks: list[RetrievedChunkDict]

    # generation
    answer: str
    citations: list[str]
    generation_raw: str             # unparsed model output, for tracing

    # validation (schema-level)
    invalid_citations: list[str]
    validation_passed: bool

    # grounding (README §8: per-sentence NLI entailment vs cited chunks)
    grounding_checked: bool
    grounding_score: Optional[float]     # fraction of sentences entailed, 0..1
    ungrounded_sentences: list[str]      # hallucination flags

    # judge
    judge_score: float              # 0..1
    judge_reason: str

    # HITL
    needs_review: bool
    review_reason: str
    reviewer_verdict: Optional[str]  # "approved" | "edited" | "rejected"
    reviewer_note: Optional[str]

    # output
    final_answer: str
    final_citations: list[str]
    status: str                     # "answered" | "rejected" | "error"
    error: Optional[str]
