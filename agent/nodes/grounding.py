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


def _verbalize_table_rows(premise: str, hypothesis: str, cap: int = 8) -> list[str]:
    """Markdown table rows → short natural-language premises.

    NLI models are poor at (markdown table → sentence) entailment, so a row
    like `| OP-3003 | MRI brain | 680.00 | 20% | Yes |` becomes
    "Procedure Code: OP-3003; Description: MRI brain; ...", which entails
    "The copay for MRI brain is 20%" reliably. Rows are lexically prefiltered
    against the hypothesis so at most ``cap`` NLI calls are added.
    """
    lines = [l.strip() for l in premise.splitlines() if l.strip().startswith("|")]
    if len(lines) < 3:
        return []
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    hyp_tokens = {t for t in _normalize(hypothesis).split() if len(t) > 2}
    rows: list[str] = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(header) or not any(cells):
            continue
        verbalized = "; ".join(f"{h}: {c}" for h, c in zip(header, cells)) + "."
        row_tokens = set(_normalize(verbalized).split())
        if hyp_tokens & row_tokens:
            rows.append(verbalized)
    return rows[:cap]


def _sentence_grounded_prob(sentence: str, premise: str) -> float:
    """Max entailment of one answer sentence against one cited chunk.

    Stages:
      1. Lexical containment fast-path — a sentence that appears verbatim in
         the premise is grounded by definition (extractive answers are the
         common case), no model call needed.
      2. Verbalized table rows (tables entail poorly as raw markdown).
      3. NLI over sliding sentence windows of the premise. Small NLI models
         dilute over long premises (observed live: a verbatim quote scored
         under 0.5 against a full 200-word chunk), so score against 3-sentence
         windows and take the max.
    """
    if _normalize(sentence) and _normalize(sentence) in _normalize(premise):
        return 1.0
    best = 0.0
    for row in _verbalize_table_rows(premise, sentence):
        best = max(best, _entailment_prob(row, sentence))
        if best >= 0.95:
            return best
    premise_sentences = [s for s in split_sentences(premise)
                         if not s.lstrip().startswith("|")]
    if not premise_sentences:
        return best
    if len(premise_sentences) <= 3:
        return max(best, _entailment_prob(" ".join(premise_sentences), sentence))
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

    premises = [c["text"][:6000] for c in cited]  # keep full tables (see generate.py)
    premise_ids = [c["chunk_id"] for c in cited]
    sentences = split_sentences(answer)
    if not sentences:
        return {"grounding_checked": False, "grounding_score": None,
                "ungrounded_sentences": []}

    # Fragment fallback: a very short answer ("20%") is a degenerate NLI
    # hypothesis, and the model often cites prose chunks while the value
    # lives in the table. Verbatim containment in ANY retrieved chunk is the
    # strongest grounding evidence available for fragments.
    all_premises = [(c["chunk_id"], c["text"][:6000]) for c in chunks]

    ungrounded: list[str] = []
    attributions: list[dict] = []  # which cited chunk grounds each sentence
    for sentence in sentences:
        # Entailed if ANY cited chunk entails it (multi-source answers).
        scores = [_sentence_grounded_prob(sentence, p) for p in premises]
        best = max(scores)
        best_chunk = premise_ids[scores.index(best)]
        if best < settings.entailment_threshold and len(sentence.split()) <= 5:
            norm = _normalize(sentence)
            for cid, text in all_premises:
                if norm and norm in _normalize(text):
                    best, best_chunk = 1.0, cid
                    logger.info("Fragment %r grounded by containment in %s",
                                sentence, cid)
                    break
        attributions.append({
            "sentence": sentence,
            "chunk_id": best_chunk if best >= settings.entailment_threshold else None,
            "prob": round(best, 3),
        })
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
        "sentence_attributions": attributions,
    }
