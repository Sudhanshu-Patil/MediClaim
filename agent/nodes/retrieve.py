"""Retrieval node: run the step-3 HybridRetriever according to the route."""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

from agent.state import AgentState
from retrieval.retriever import HybridRetriever

logger = logging.getLogger(__name__)

_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    # Process-level singleton: embedding + reranker models load once.
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def retrieve(state: AgentState) -> dict:
    route = state.get("route", "hybrid")
    results = get_retriever().retrieve(
        state["query"],
        source_type=state.get("source_type"),
        use_graph=route in ("graph", "hybrid"),
    )
    chunks = [dataclasses.asdict(r) for r in results]
    logger.info("Retrieved %d chunks (route=%s)", len(chunks), route)
    return {"chunks": chunks}
