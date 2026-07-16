"""LLM-as-judge node: pre-response faithfulness/relevance scoring (README §4).

A second call to the same local model, in judge persona, scoring the drafted
answer against its cited chunks. The score feeds the HITL gate: low scores
route to human review instead of going out the door.
"""

from __future__ import annotations

import json
import logging

from agent.nodes.generate import get_llm
from agent.state import AgentState

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = (
    "You are a strict quality judge for a claims-adjudication assistant. "
    "Given a question, the source excerpts, and a drafted answer, score the "
    "answer. Respond ONLY with JSON: "
    '{"faithfulness": <0-10>, "relevance": <0-10>, "reason": "<one sentence>"}. '
    "faithfulness = is every claim in the answer supported by the excerpts; "
    "relevance = does it actually answer the question."
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
        raw = get_llm().chat(
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": user}],
            json_mode=True, temperature=0.0, max_tokens=200,
        )
        obj = json.loads(raw)
        faith = float(obj.get("faithfulness", 0))
        rel = float(obj.get("relevance", 0))
        score = round(min(faith, rel) / 10.0, 3)  # weakest-link, normalized 0..1
        reason = str(obj.get("reason", ""))
    except Exception as exc:  # judge failure must not kill the pipeline
        logger.exception("Judge call failed; scoring 0 to force review")
        score, reason = 0.0, f"judge error: {exc}"
    logger.info("Judge score %.2f (%s)", score, reason[:80])
    return {"judge_score": score, "judge_reason": reason}
