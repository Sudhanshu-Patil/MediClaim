"""Build the QLoRA instruction dataset (roadmap step 1).

Two example families, mixed:

1. **Citation-format RAG examples** (generated from the chunks actually in
   Qdrant): the user turn contains retrieved context blocks tagged with their
   real ``chunk_id``s and a question; the assistant turn is a JSON object
   ``{"answer": ..., "citations": [chunk_id, ...]}``. This trains the exact
   schema-constrained, structurally-cited generation the architecture depends
   on (README §3 "citations are structural, not post-hoc", §8). Includes
   refusal examples where the context cannot answer the question.

2. **Domain-knowledge examples** from MedQuad (public medical QA, HF Hub):
   plain question → answer, no context and no citations — teaching the model
   to cite when context is given and answer plainly when it isn't.

Output: chat-format JSONL (``{"messages": [...]}``) ready for TRL's
SFTTrainer, split into train/val. Run locally (needs the Qdrant stack up);
commit the JSONL so the Colab notebook can train from the repo.

    python finetuning/build_dataset.py --medquad 2000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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

# Questions the sample policy cannot answer — refusal training.
_OFF_CONTEXT_QUESTIONS = [
    "What is the reimbursement rate for inpatient cardiac surgery?",
    "Does this policy cover dental implants?",
    "What is the annual out-of-pocket maximum for a family plan?",
    "How are emergency ambulance services reimbursed?",
    "What is the copay for a kidney transplant evaluation?",
    "Are hearing aids covered under this policy?",
]

_TABLE_ROW_RE = re.compile(r"^\|(.+)\|\s*$")


def _fetch_chunks() -> list[dict]:
    from retrieval.vector_store import QdrantStore

    store = QdrantStore()
    points, _ = store.client.scroll(
        collection_name=store.collection,
        scroll_filter=None,
        limit=1000,
        with_payload=True,
    )
    active = [p.payload for p in points if p.payload.get("status") == "active"]
    logger.info("Fetched %d active chunks from Qdrant", len(active))
    return active


def _context_block(chunk_id: str, text: str, max_chars: int = 3500) -> str:
    return f"[chunk_id={chunk_id}]\n{text[:max_chars]}"


def _example(context_chunks: list[tuple[str, str]], question: str,
             answer: str, citations: list[str]) -> dict:
    context = "\n\n".join(_context_block(cid, text) for cid, text in context_chunks)
    user = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    assistant = json.dumps({"answer": answer, "citations": citations})
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _parse_table_rows(markdown: str) -> tuple[list[str], list[list[str]]]:
    rows = []
    for line in markdown.splitlines():
        m = _TABLE_ROW_RE.match(line.strip())
        if not m:
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if cells and not set("".join(cells)) <= {"-", ":", " "}:
            rows.append(cells)
    if not rows:
        return [], []
    return rows[0], rows[1:]


def build_citation_examples(chunks: list[dict], rng: random.Random) -> list[dict]:
    examples: list[dict] = []
    children = [c for c in chunks if c.get("chunk_type") == "child"]
    tables = [c for c in chunks if c.get("chunk_type") == "table"]

    # ── Table row lookups (the flagship behavior: read structured rows) ────
    for table in tables:
        header, rows = _parse_table_rows(table.get("text", ""))
        if len(header) < 5:
            continue
        cid = table["chunk_id"]
        for cells in rows:
            if len(cells) < 5:
                continue
            code, desc, benefit, copay, auth = cells[:5]
            auth_answer = (
                f"Yes, prior authorization is required for {desc.lower()} ({code})."
                if auth.lower().startswith("y")
                else f"No, prior authorization is not required for {desc.lower()} ({code})."
            )
            for question, answer in [
                (f"What is the copay for {desc.lower()}?",
                 f"The member copay for {desc.lower()} ({code}) is {copay}."),
                (f"Is prior authorization required for {desc.lower()}?", auth_answer),
                (f"What is the maximum benefit for {desc.lower()}?",
                 f"The maximum benefit for {desc.lower()} ({code}) is {benefit} USD per occurrence."),
            ]:
                # Distractor context chunk teaches selective citation.
                distractor = rng.choice(children) if children else None
                ctx = [(cid, table["text"])]
                if distractor:
                    ctx.append((distractor["chunk_id"], distractor["text"]))
                    rng.shuffle(ctx)
                examples.append(_example(ctx, question, answer, [cid]))

    # ── Section prose QA ────────────────────────────────────────────────────
    for chunk in children:
        title = (chunk.get("section_title") or "this section").strip()
        cid = chunk["chunk_id"]
        examples.append(
            _example(
                [(cid, chunk["text"])],
                f"What does the policy say in section '{title}'?",
                chunk["text"][:800],
                [cid],
            )
        )

    # ── Refusals: context present but does not answer ───────────────────────
    for question in _OFF_CONTEXT_QUESTIONS:
        if not children:
            break
        chunk = rng.choice(children)
        examples.append(
            _example(
                [(chunk["chunk_id"], chunk["text"])],
                question,
                "The provided context does not contain information to answer "
                "this question. Please retrieve the relevant policy section "
                "or route the claim to manual review.",
                [],
            )
        )
    logger.info("Built %d citation-format examples", len(examples))
    return examples


def build_medquad_examples(limit: int, rng: random.Random) -> list[dict]:
    if limit <= 0:
        return []
    from datasets import load_dataset

    ds = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
    indices = rng.sample(range(len(ds)), min(limit, len(ds)))
    examples = []
    for i in indices:
        row = ds[i]
        question = (row.get("Question") or "").strip()
        answer = (row.get("Answer") or "").strip()
        if not question or not answer:
            continue
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer[:2000]},
                ]
            }
        )
    logger.info("Built %d MedQuad examples", len(examples))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medquad", type=int, default=2000,
                        help="number of MedQuad domain examples to mix in")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="finetuning/data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    rng = random.Random(args.seed)

    examples = build_citation_examples(_fetch_chunks(), rng)
    examples += build_medquad_examples(args.medquad, rng)
    rng.shuffle(examples)

    n_val = max(1, int(len(examples) * args.val_fraction))
    val, train = examples[:n_val], examples[n_val:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, split in [("train.jsonl", train), ("val.jsonl", val)]:
        with open(out / name, "w", encoding="utf-8") as fh:
            for ex in split:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(train)} train / {len(val)} val examples to {out}/")


if __name__ == "__main__":
    main()
