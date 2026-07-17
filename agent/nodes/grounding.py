"""Grounding node: per-sentence NLI entailment against cited chunks.

README §8 point 2 — the *live* guardrail, distinct from offline RAGAS eval:
every sentence of the generated answer must be entailed by at least one of
the chunks the generator cited. Sentences that aren't are flagged
(hallucination flagging, README §10), and a low grounded fraction routes the
answer to human review via the risk gate.

Model: a small local NLI cross-encoder (default
cross-encoder/nli-deberta-v3-xsmall, ~280 MB RAM, CPU). Lazy-loaded;
GROUNDING_ENABLED=0 or a load failure degrades gracefully — the answer is
marked unchecked (grounding_checked=false) rather than blocking the pipeline,
and the judge/HITL layers still stand behind it.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from config import get_settings
from agent.state import AgentState

logger = logging.getLogger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

_model = None
_tokenizer = None
_entail_idx: Optional[int] = None
_load_failed = False


def _get_nli():
    global _model, _tokenizer, _entail_idx, _load_failed
    if _model is None and not _load_failed:
        settings = get_settings()
        try:
            import torch  # noqa: F401  (present via docling)
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)

            name = settings.grounding_model
            logger.info("Loading NLI grounding model %s", name)
            _tokenizer = AutoTokenizer.from_pretrained(name)
            _model = AutoModelForSequenceClassification.from_pretrained(name)
            _model.eval()
            # Find the "entailment" logit index from config, never hardcode.
            id2label = {int(k): v.lower() for k, v in _model.config.id2label.items()}
            _entail_idx = next(i for i, l in id2label.items() if "entail" in l)
        except Exception:
            _load_failed = True
            logger.exception(
                "NLI model failed to load — grounding checks disabled "
                "(answers will be marked grounding_checked=false)"
            )
    return _model


def _entailment_prob(premise: str, hypothesis: str) -> float:
    import torch

    inputs = _tokenizer(
        premise, hypothesis, truncation=True, max_length=512, return_tensors="pt"
    )
    with torch.no_grad():
        logits = _get_nli()(**inputs).logits[0]
    return float(torch.softmax(logits, dim=-1)[_entail_idx])


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text) if len(s.strip()) > 2]


_NORM_RE = re.compile(r"[^a-z0-9%$. ]+")


def _normalize(text: str) -> str:
    return _NORM_RE.sub(" ", text.lower()).replace("  ", " ").strip()


def _sentence_grounded_prob(sentence: str, premise: str) -> float:
    """Max entailment of one answer sentence against one cited chunk.

    Two-stage:
      1. Lexical containment fast-path — a sentence that appears verbatim in
         the premise is grounded by definition (extractive answers are the
         common case), no model call needed.
      2. NLI over sliding sentence windows of the premise. Small NLI models
         dilute over long premises (observed live: a verbatim quote scored
         under 0.5 against a full 200-word chunk), so score against 3-sentence
         windows and take the max.
    """
    if _normalize(sentence) and _normalize(sentence) in _normalize(premise):
        return 1.0
    premise_sentences = split_sentences(premise)
    if len(premise_sentences) <= 3:
        return _entailment_prob(premise, sentence)
    best = 0.0
    for i in range(0, len(premise_sentences), 2):
        window = " ".join(premise_sentences[i : i + 3])
        best = max(best, _entailment_prob(window, sentence))
        if best >= 0.95:
            break
    return best


def grounding(state: AgentState) -> dict:
    settings = get_settings()
    answer = state.get("answer", "")
    chunks = state.get("chunks", [])
    cited_ids = set(state.get("citations", []))
    cited = [c for c in chunks if c["chunk_id"] in cited_ids]

    # Nothing to check: refusals/uncited answers are the validator's problem.
    if not settings.grounding_enabled or not answer or not cited:
        return {"grounding_checked": False, "grounding_score": None,
                "ungrounded_sentences": []}
    if _get_nli() is None:
        return {"grounding_checked": False, "grounding_score": None,
                "ungrounded_sentences": []}

    premises = [c["text"][:3000] for c in cited]
    sentences = split_sentences(answer)
    if not sentences:
        return {"grounding_checked": False, "grounding_score": None,
                "ungrounded_sentences": []}

    ungrounded: list[str] = []
    for sentence in sentences:
        # Entailed if ANY cited chunk entails it (multi-source answers).
        best = max(_sentence_grounded_prob(sentence, p) for p in premises)
        if best < settings.entailment_threshold:
            ungrounded.append(sentence)

    score = round(1.0 - len(ungrounded) / len(sentences), 3)
    if ungrounded:
        logger.warning(
            "Grounding: %d/%d sentences NOT entailed by cited chunks: %s",
            len(ungrounded), len(sentences),
            [s[:60] for s in ungrounded],
        )
    else:
        logger.info("Grounding: all %d sentences entailed", len(sentences))
    return {
        "grounding_checked": True,
        "grounding_score": score,
        "ungrounded_sentences": ungrounded,
    }
