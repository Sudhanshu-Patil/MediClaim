"""Input guardrails node (README §10): PII redaction + jailbreak/toxicity.

Runs before anything else touches the query:

  * **PII redaction** — deterministic regexes for the identifiers that show up
    in claims workflows (SSN, member/claim IDs, credit cards with Luhn check,
    phone numbers, emails, dates of birth). Redacted, not blocked: the query
    still runs, but PII never reaches retrieval logs, LLM prompts, or traces.
  * **Jailbreak / prompt-injection** — pattern screen for instruction-override
    attempts. Blocked outright (no LLM call is ever made).
  * **Toxicity** — small lexicon screen; blocked.

Implementation note: these are lightweight, deterministic, zero-cost local
checks in the spirit of the stack (the README names Guardrails AI / NeMo
Guardrails; both pull heavyweight dependencies — presidio/spacy or an LLM in
the loop. This module is the laptop-friendly implementation of the same
contract, and the node boundary makes swapping a library in later trivial).
"""

from __future__ import annotations

import logging
import re

from agent.state import AgentState

logger = logging.getLogger(__name__)

# ── PII patterns (order matters: most specific first) ───────────────────────
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("credit_card", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")),
    ("phone", re.compile(r"\b(?:\+?\d{1,3}[ -]?)?(?:\(\d{3}\)|\d{3})[ -]?\d{3}[ -]?\d{4}\b")),
    ("member_id", re.compile(r"\b(?:member|policy|claim|subscriber)\s*(?:id|no|number|#)\s*[:#]?\s*[A-Z0-9-]{6,}\b", re.IGNORECASE)),
    ("dob", re.compile(r"\b(?:dob|date\s+of\s+birth)\s*[:#]?\s*\d{1,4}[/-]\d{1,2}[/-]\d{1,4}\b", re.IGNORECASE)),
]


def _luhn_ok(digits: str) -> bool:
    ds = [int(c) for c in re.sub(r"\D", "", digits)]
    if len(ds) < 13:
        return False
    total, parity = 0, len(ds) % 2
    for i, d in enumerate(ds):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII with [REDACTED:<kind>]; returns (clean_text, kinds_found)."""
    found: list[str] = []
    for kind, pattern in _PII_PATTERNS:
        def _sub(match: re.Match, kind=kind) -> str:
            # Credit-card pattern is broad — only redact if it Luhn-validates,
            # so claim amounts like "1850.00" or long codes survive.
            if kind == "credit_card" and not _luhn_ok(match.group(0)):
                return match.group(0)
            found.append(kind)
            return f"[REDACTED:{kind}]"

        text = pattern.sub(_sub, text)
    return text, sorted(set(found))


# ── Jailbreak / prompt-injection patterns ────────────────────────────────────
_JAILBREAK_RE = re.compile(
    r"(ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts?)|"
    r"disregard\s+(?:your|the)\s+(?:instructions|system\s+prompt|rules)|"
    r"reveal\s+(?:your|the)\s+system\s+prompt|"
    r"you\s+are\s+now\s+(?:DAN|unrestricted|jailbroken)|"
    r"pretend\s+(?:you\s+have\s+no|there\s+are\s+no)\s+(?:rules|restrictions|guidelines)|"
    r"act\s+as\s+an?\s+(?:unfiltered|uncensored)|"
    r"do\s+anything\s+now|"
    r"output\s+the\s+(?:full|entire)\s+(?:system|hidden)\s+prompt)",
    re.IGNORECASE,
)

_TOXICITY_RE = re.compile(
    r"\b(kill\s+(?:yourself|him|her|them)|how\s+to\s+(?:harm|hurt|poison)\s+|"
    r"\bslur\b|racial\s+slur)\b",
    re.IGNORECASE,
)


def input_guard(state: AgentState) -> dict:
    query = state["query"]

    if _JAILBREAK_RE.search(query):
        logger.warning("Input blocked: jailbreak/prompt-injection pattern")
        return {
            "input_blocked": True,
            "input_block_reason": "jailbreak/prompt-injection pattern detected",
        }
    if _TOXICITY_RE.search(query):
        logger.warning("Input blocked: toxicity pattern")
        return {
            "input_blocked": True,
            "input_block_reason": "toxic content detected",
        }

    clean, kinds = redact_pii(query)
    if kinds:
        logger.info("PII redacted from query: %s", kinds)
        return {
            "input_blocked": False,
            "query": clean,           # downstream nodes only ever see the clean text
            "pii_redacted": kinds,
        }
    return {"input_blocked": False, "pii_redacted": []}
