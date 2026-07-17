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
import re
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


def build_user_message(query: str, chunks: list[dict], max_chunk_chars: int = 6000) -> str:
    # 6000 (not 3500): the 46-row reimbursement table is ~4400 chars; the old
    # cap silently cut every row after OP-6003, making the model confidently
    # wrong about anything in the table's tail (caught by NLI grounding).
    # Requires num_ctx 8192 in the Modelfile.
    blocks = [
        f"[chunk_id={c['chunk_id']}]\n{c['text'][:max_chunk_chars]}"
        for c in chunks
    ]
    # NOTE: keep this EXACTLY in the fine-tuned format (finetuning/
    # build_dataset.py). A tested "answer in complete sentences" nudge after
    # the question derailed the model into echoing the context verbatim —
    # the fine-tune is format-locked; richer answers are a v2 training-data
    # change, not an inference-time prompt tweak.
    return "CONTEXT:\n" + "\n\n".join(blocks) + f"\n\nQUESTION: {query}"


def _normalize_citation(citation: str) -> str:
    """The 3B model sometimes emits "chunk_id=<uuid>" instead of the bare id."""
    citation = citation.strip().strip("[]")
    if "=" in citation:
        citation = citation.split("=", 1)[1]
    return citation.strip()


_INLINE_CHUNK_REF_RE = re.compile(r"\s*\[chunk_id=[0-9a-f-]+\]\s*")


def parse_generation(raw: str) -> tuple[str, list[str]]:
    """Parse {"answer", "citations"} — salvage what we can from bad JSON."""
    try:
        obj = json.loads(raw)
        answer = str(obj.get("answer", "")).strip()
        # Double-wrapped JSON (observed in hard-case testing): the answer
        # value is itself a serialized {"answer": ...} object — unwrap once.
        if answer.startswith('{"answer"'):
            try:
                inner = json.loads(answer)
                answer = str(inner.get("answer", answer)).strip()
            except json.JSONDecodeError:
                pass
        # Strip chunk-id references the model sometimes appends inline.
        answer = _INLINE_CHUNK_REF_RE.sub(" ", answer).strip()
        citations = [_normalize_citation(str(c)) for c in obj.get("citations", []) if c]
        if answer:
            return answer, [c for c in citations if c]
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass
    # Not valid JSON: treat the whole output as the answer with no citations —
    # downstream validation will flag it and route to review.
    return raw.strip(), []


class AnswerFieldStreamer:
    """Incrementally extract the "answer" string value from streamed JSON.

    The model emits {"answer": "...", "citations": [...]} token by token; the
    UI should render the prose as it generates, never the raw JSON. Feed each
    delta in; it returns only the characters inside the answer value
    (JSON-unescaped), going silent once the closing quote arrives.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._in_answer = False
        self._done = False
        self._escaped = False

    def feed(self, delta: str) -> str:
        if self._done:
            return ""
        out = []
        for ch in delta:
            if not self._in_answer:
                self._buffer += ch
                if re.search(r'"answer"\s*:\s*"$', self._buffer):
                    self._in_answer = True
                continue
            if self._escaped:
                out.append({"n": "\n", "t": "\t", '"': '"', "\\": "\\"}.get(ch, ch))
                self._escaped = False
            elif ch == "\\":
                self._escaped = True
            elif ch == '"':
                self._done = True
                break
            else:
                out.append(ch)
        return "".join(out)


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

    # When invoked via app.stream(..., stream_mode="custom") (the API path),
    # answer tokens stream out live; otherwise the blocking call is used.
    writer = None
    try:
        from langgraph.config import get_stream_writer

        writer = get_stream_writer()
    except Exception:
        pass

    # temperature 0: adjudication answers should be deterministic — same
    # policy + same question must yield the same answer (and it makes eval
    # runs reproducible).
    llm = get_llm()
    # 400 tokens caps runaway repetition loops (a correct answer here is
    # 1-3 sentences; 1024 let a degenerate loop burn ~60s of GPU time).
    if writer is not None and hasattr(llm, "chat_stream"):
        streamer = AnswerFieldStreamer()
        parts: list[str] = []
        for delta in llm.chat_stream(messages, json_mode=True, temperature=0.0,
                                     max_tokens=400):
            parts.append(delta)
            visible = streamer.feed(delta)
            if visible:
                writer({"token": visible})
        raw = "".join(parts)
    else:
        raw = llm.chat(messages, json_mode=True, temperature=0.0, max_tokens=400)

    answer, citations = parse_generation(raw)
    logger.info("Generated answer (%d chars, %d citations)", len(answer), len(citations))
    return {"answer": answer, "citations": citations, "generation_raw": raw}
