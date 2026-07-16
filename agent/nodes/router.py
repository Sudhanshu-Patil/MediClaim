"""Router node: decide vector / graph / hybrid retrieval per query (README §3).

Deliberately rule-based, not an LLM call: routing runs on every query, so it
must be fast and deterministic. Relationship- or reference-shaped questions
lean on the graph; short factual lookups are vector-friendly; everything else
takes the hybrid path (both branches + RRF), which is also the safe default.
"""

from __future__ import annotations

import logging
import re

from agent.state import AgentState

logger = logging.getLogger(__name__)

_GRAPH_HINTS = re.compile(
    r"\b(relat(?:ed|ionship)|refer(?:s|ence|enced)?|cross[- ]?ref|section\s+\d|"
    r"appendix|supersed|version|history|between|connect(?:ed|ion)|depends?\s+on|"
    r"linked)\b",
    re.IGNORECASE,
)
_LOOKUP_HINTS = re.compile(
    r"\b(copay|co-pay|benefit|rate|amount|cost|price|code|OP-\d+|deductible|"
    r"limit|maximum|how\s+much)\b",
    re.IGNORECASE,
)


def route_query(state: AgentState) -> dict:
    query = state["query"]
    if _GRAPH_HINTS.search(query):
        route = "hybrid"      # graph signal present — keep vector too, fuse
    elif _LOOKUP_HINTS.search(query) and len(query.split()) <= 12:
        route = "vector"      # short factual lookup; skip graph latency
    else:
        route = "hybrid"
    logger.info("Routed %r -> %s", query[:60], route)
    return {"route": route}
