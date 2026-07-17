"""LangGraph agent assembly (roadmap step 4, architecture QUERY subgraph).

    router → retrieve → generate → validate → judge ──┬─ ok ──────→ finalize
                                                      └─ high-risk → hitl_review
                                                                     (interrupt)
                                                                        ↓
                                                                     finalize

HITL (README §10): `interrupt()` pauses the graph when the answer failed
schema validation, the judge scored it low, or the query itself is
risk-flagged. State checkpoints to Redis (falls back to in-memory when the
Redis checkpointer package isn't installed), so a paused review survives
process restarts and is resumed by thread_id with the reviewer's verdict.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agent.nodes.generate import generate
from agent.nodes.judge import judge
from agent.nodes.retrieve import retrieve
from agent.nodes.router import route_query
from agent.nodes.validate import validate
from agent.state import AgentState

logger = logging.getLogger(__name__)

JUDGE_THRESHOLD = float(os.getenv("JUDGE_THRESHOLD", "0.6"))

# Queries that are inherently high-stakes go to review even with a good score.
_HIGH_RISK_RE = re.compile(
    r"\b(den(?:y|ial|ied)|fraud|terminat|overrid|exceed|dispute|lawsuit|appeal\s+decision)\b",
    re.IGNORECASE,
)


def risk_gate(state: AgentState) -> dict:
    reasons = []
    if not state.get("validation_passed", True):
        reasons.append("citation validation failed")
    if state.get("judge_score", 0.0) < JUDGE_THRESHOLD:
        reasons.append(f"judge score {state.get('judge_score', 0.0):.2f} < {JUDGE_THRESHOLD}")
    if _HIGH_RISK_RE.search(state.get("query", "")):
        reasons.append("high-risk query pattern")
    needs_review = bool(reasons)
    return {"needs_review": needs_review, "review_reason": "; ".join(reasons)}


def hitl_review(state: AgentState) -> dict:
    """Pause for a human. interrupt() raises; on resume it returns the verdict.

    The resume payload is a dict: {"verdict": "approved"|"edited"|"rejected",
    "note": str, "answer": str (when edited)}.
    """
    decision = interrupt(
        {
            "reason": state.get("review_reason", ""),
            "query": state.get("query", ""),
            "draft_answer": state.get("answer", ""),
            "citations": state.get("citations", []),
            "judge_score": state.get("judge_score"),
            "judge_reason": state.get("judge_reason", ""),
        }
    )
    verdict = str(decision.get("verdict", "rejected"))
    update: dict = {
        "reviewer_verdict": verdict,
        "reviewer_note": decision.get("note"),
    }
    if verdict == "edited" and decision.get("answer"):
        update["answer"] = str(decision["answer"])
        if decision.get("citations") is not None:
            update["citations"] = list(decision["citations"])
    return update


def finalize(state: AgentState) -> dict:
    if state.get("reviewer_verdict") == "rejected":
        return {
            "final_answer": "This response was rejected by a human reviewer."
            + (f" Note: {state['reviewer_note']}" if state.get("reviewer_note") else ""),
            "final_citations": [],
            "status": "rejected",
        }
    # Strip citations that failed validation rather than shipping bad ones.
    invalid = set(state.get("invalid_citations", []))
    citations = [c for c in state.get("citations", []) if c not in invalid]
    return {
        "final_answer": state.get("answer", ""),
        "final_citations": citations,
        "status": "answered",
    }


def _after_risk_gate(state: AgentState) -> str:
    return "hitl_review" if state.get("needs_review") else "finalize"


def _make_checkpointer():
    """Persistence tiers for HITL state (README §6 role 4).

    1. Redis (langgraph-checkpoint-redis; needs Redis Stack modules) — the
       target architecture.
    2. SQLite file — zero-infra persistence; paused reviews still survive
       process restarts, so the CLI --resume flow works.
    3. In-memory — last resort, same-process resume only.
    """
    try:
        from langgraph.checkpoint.redis import RedisSaver  # langgraph-checkpoint-redis

        from config import get_settings

        saver = RedisSaver.from_conn_string(get_settings().redis_url)
        saver.setup()
        logger.info("Using Redis checkpointer")
        return saver
    except Exception:
        pass
    try:
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver  # langgraph-checkpoint-sqlite

        path = os.getenv("CHECKPOINT_SQLITE_PATH", ".langgraph_checkpoints.sqlite")
        conn = sqlite3.connect(path, check_same_thread=False)
        logger.info("Using SQLite checkpointer at %s", path)
        return SqliteSaver(conn)
    except Exception:
        from langgraph.checkpoint.memory import MemorySaver

        logger.info("No persistent checkpointer available; using in-memory "
                    "(dev only — resume must happen in the same process)")
        return MemorySaver()


def build_graph(checkpointer=None):
    from observability.tracing import traced

    graph = StateGraph(AgentState)
    # Each node becomes a Langfuse span (identity wrapper when tracing is off).
    graph.add_node("router", traced("router")(route_query))
    graph.add_node("retrieve", traced("retrieve")(retrieve))
    graph.add_node("generate", traced("generate")(generate))
    graph.add_node("validate", traced("validate")(validate))
    graph.add_node("judge", traced("judge")(judge))
    graph.add_node("risk_gate", traced("risk_gate")(risk_gate))
    graph.add_node("hitl_review", hitl_review)  # interrupt() must not be wrapped
    graph.add_node("finalize", traced("finalize")(finalize))

    graph.add_edge(START, "router")
    graph.add_edge("router", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "validate")
    graph.add_edge("validate", "judge")
    graph.add_edge("judge", "risk_gate")
    graph.add_conditional_edges("risk_gate", _after_risk_gate,
                                {"hitl_review": "hitl_review", "finalize": "finalize"})
    graph.add_edge("hitl_review", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer or _make_checkpointer())


def run_agent(app, payload, config) -> dict:
    """Invoke the graph under one root Langfuse trace (README §11).

    Wraps app.invoke so every node span and LLM generation nests under a
    single "medclaim-agent" trace, attaches route/status metadata, records
    the judge score as a trace score, and flushes the batch queue (required
    in short-lived CLI processes). Identical to app.invoke when tracing is
    off.
    """
    from observability import tracing

    if not tracing.enabled():
        return app.invoke(payload, config)

    @tracing.traced("medclaim-agent")
    def _run() -> dict:
        state = app.invoke(payload, config)
        tracing.update_trace(
            metadata={
                "thread_id": config.get("configurable", {}).get("thread_id"),
                "route": state.get("route"),
                "status": state.get("status"),
                "needs_review": state.get("needs_review"),
                "interrupted": "__interrupt__" in state,
                "n_chunks": len(state.get("chunks", [])),
            },
            tags=["agent"],
        )
        if state.get("judge_score") is not None:
            tracing.score_trace(
                "judge_score", float(state["judge_score"]),
                comment=state.get("judge_reason"),
            )
        if state.get("validation_passed") is not None:
            tracing.score_trace(
                "citation_validation", 1.0 if state["validation_passed"] else 0.0
            )
        return state

    try:
        return _run()
    finally:
        tracing.flush()
