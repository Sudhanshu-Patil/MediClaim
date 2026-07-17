"""Validation node: citations must resolve to actually-retrieved chunk_ids.

This is the schema half of the output guardrail (README §10: "citations must
resolve to real chunk_ids"). The NLI per-sentence entailment check is a
separate node in step 5 — this one is cheap and runs always.
"""

from __future__ import annotations

import logging

from agent.state import AgentState

logger = logging.getLogger(__name__)


_ECHO_MARKERS = ("CONTEXT:", "[chunk_id=")


def validate(state: AgentState) -> dict:
    retrieved_ids = {c["chunk_id"] for c in state.get("chunks", [])}
    citations = state.get("citations", [])
    invalid = [c for c in citations if c not in retrieved_ids]

    # Empty answer (e.g. the model answered with only a chunk_id, stripped by
    # parsing): nothing to ship — route to review for a human-written answer.
    answer_text = state.get("answer", "")
    if not answer_text.strip():
        logger.warning("Validation failed: empty generation")
        return {"invalid_citations": invalid, "validation_passed": False}

    # Generation format failure: the format-locked model occasionally echoes
    # the CONTEXT block instead of answering. Never let that ship as prose.
    if any(m in answer_text for m in _ECHO_MARKERS):
        logger.warning("Validation failed: generation echoed context/markup")
        return {"invalid_citations": invalid, "validation_passed": False}

    # Degenerate repetition: the same 6-word shingle recurring 3+ times means
    # the model looped (observed live at temperature 0). Flag, don't ship.
    words = answer_text.lower().split()
    if len(words) >= 30:
        shingles: dict[tuple, int] = {}
        for i in range(len(words) - 5):
            key = tuple(words[i:i + 6])
            shingles[key] = shingles.get(key, 0) + 1
        if max(shingles.values()) >= 3:
            logger.warning("Validation failed: degenerate repetitive generation")
            return {"invalid_citations": invalid, "validation_passed": False}

    # An answer with no citations is only valid if it says it can't answer —
    # heuristically: refusal-ish wording. Otherwise it's an ungrounded claim.
    passed = not invalid
    if not citations and state.get("chunks"):
        answer_lower = state.get("answer", "").lower()
        refusal_markers = ("does not contain", "cannot answer", "no information",
                           "not able to answer", "manual review")
        if not any(m in answer_lower for m in refusal_markers):
            passed = False

    if not passed:
        logger.warning("Validation failed: invalid=%s, citations=%s", invalid, citations)
    return {"invalid_citations": invalid, "validation_passed": passed}
