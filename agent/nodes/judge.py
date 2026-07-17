"""LLM-as-judge node: pre-response faithfulness/relevance scoring (README §4).

A second call to the same local model, in judge persona, scoring the drafted
answer against its cited chunks. The score feeds the HITL gate: low scores
route to human review instead of going out the door.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from agent.llm_client import OllamaClient
from agent.nodes.generate import get_llm
from agent.state import AgentState

logger = logging.getLogger(__name__)

_judge_llm: Optional[OllamaClient] = None


def get_judge_llm() -> OllamaClient:
    """Judge model, separately configurable (LLM_JUDGE_MODEL).

    Defaults to the generator model. Known limitation (seen in Langfuse
    traces): a 3B judge scores erratically — e.g. zero-scoring a verbatim
    quote for "lacking additional context". Point LLM_JUDGE_MODEL at a
    stronger local model when available; the dedicated NLI grounding check
    (roadmap step 5) is the principled fix for faithfulness.
    """
    global _judge_llm
    if _judge_llm is None:
        judge_model = os.getenv("LLM_JUDGE_MODEL")
        _judge_llm = OllamaClient(model=judge_model) if judge_model else get_llm()
    return _judge_llm

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_judge_json(raw: str) -> dict:
    """Tolerant parse: a small local judge sometimes malforms its JSON.

    Observed in traces: {"reason": unquoted text} — invalid JSON that no
    loads() can fix. Fall back to field-level regex extraction; the numeric
    scores are all the gate actually needs.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_OBJECT_RE.search(raw)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    salvaged: dict = {}
    for key in ("faithfulness", "relevance"):
        m = re.search(rf'"{key}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw)
        if m:
            salvaged[key] = float(m.group(1))
    m = re.search(r'"reason"\s*:\s*"?([^"}]+)', raw)
    if m:
        salvaged["reason"] = m.group(1).strip()
    if salvaged:
        logger.warning("Judge JSON malformed; salvaged fields %s", sorted(salvaged))
        return salvaged
    raise ValueError(f"unparseable judge output: {raw[:120]}")

JUDGE_SYSTEM = (
    "You are a strict quality judge for a claims-adjudication assistant. "
    "Given a question, the source excerpts, and a drafted answer, score the "
    "answer. Respond ONLY with JSON, all string values in double quotes: "
    '{"faithfulness": <integer 0-10>, "relevance": <integer 0-10>, '
    '"reason": "<one sentence>"}. '
    "faithfulness: 10 = every claim is fully supported by the excerpts, "
    "0 = contradicted or unsupported. "
    "relevance: 10 = completely answers the question, 0 = off-topic. "
    "High scores mean a GOOD answer."
)


def judge(state: AgentState) -> dict:
    chunks = state.get("chunks", [])
    cited = [c for c in chunks if c["chunk_id"] in set(state.get("citations", []))]
    excerpts = "\n\n".join(f"- {c['text'][:1200]}" for c in (cited or chunks[:3]))
    user = (
        f"QUESTION: {state['query']}\n\nSOURCE EXCERPTS:\n{excerpts}\n\n"
        f"DRAFTED ANSWER: {state.get('answer','')}"
    )
    try:
        raw = get_judge_llm().chat(
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": user}],
            json_mode=True, temperature=0.0, max_tokens=200,
        )
        obj = _parse_judge_json(raw)
        faith = float(obj.get("faithfulness", 0))
        rel = float(obj.get("relevance", 0))
        score = round(min(faith, rel) / 10.0, 3)  # weakest-link, normalized 0..1
        reason = str(obj.get("reason", ""))
    except Exception as exc:  # judge failure must not kill the pipeline
        logger.exception("Judge call failed; scoring 0 to force review")
        score, reason = 0.0, f"judge error: {exc}"
    logger.info("Judge score %.2f (%s)", score, reason[:80])
    return {"judge_score": score, "judge_reason": reason}
