"""Generator node: schema-constrained answer with structural citations.

Builds the user turn in EXACTLY the format the model was fine-tuned on
(finetuning/build_dataset.py): CONTEXT blocks tagged [chunk_id=...] followed
by the question, expecting JSON {"answer": ..., "citations": [chunk_id...]}.
Provenance is captured at generation time, not reconstructed afterward
(README §3 "citations are structural, not post-hoc").
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from agent.llm_client import OllamaClient
from agent.state import AgentState

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are MedClaim, an assistant for insurance claims adjudicators and "
    "healthcare providers. When the user message contains CONTEXT blocks, "
    "answer ONLY from that context and respond with a JSON object: "
    '{"answer": "<your answer>", "citations": ["<chunk_id>", ...]} where '
    "citations lists the chunk_id of every context block you used. If the "
    "context cannot answer the question, say so in the answer and return an "
    "empty citations list. When there is no CONTEXT, answer from general "
    "medical and insurance knowledge in plain text."
)

_llm: Optional[OllamaClient] = None


def get_llm() -> OllamaClient:
    global _llm
    if _llm is None:
        _llm = OllamaClient()
    return _llm


def build_user_message(query: str, chunks: list[dict], max_chunk_chars: int = 3500) -> str:
    blocks = [
        f"[chunk_id={c['chunk_id']}]\n{c['text'][:max_chunk_chars]}"
        for c in chunks
    ]
    return "CONTEXT:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {query}"


def parse_generation(raw: str) -> tuple[str, list[str]]:
    """Parse {"answer", "citations"} — salvage what we can from bad JSON."""
    try:
        obj = json.loads(raw)
        answer = str(obj.get("answer", "")).strip()
        citations = [str(c) for c in obj.get("citations", []) if c]
        if answer:
            return answer, citations
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    # Not valid JSON: treat the whole output as the answer with no citations —
    # downstream validation will flag it and route to review.
    return raw.strip(), []


def generate(state: AgentState) -> dict:
    chunks = state.get("chunks", [])
    if not chunks:
        return {
            "answer": "No relevant policy content was retrieved for this question.",
            "citations": [],
            "generation_raw": "",
        }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(state["query"], chunks)},
    ]
    raw = get_llm().chat(messages, json_mode=True)
    answer, citations = parse_generation(raw)
    logger.info("Generated answer (%d chars, %d citations)", len(answer), len(citations))
    return {"answer": answer, "citations": citations, "generation_raw": raw}
