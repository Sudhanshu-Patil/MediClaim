"""Build the golden evaluation set from the currently ingested policy.

Emits eval/golden.jsonl — one line per sample:
    {"question", "reference" (ground-truth answer),
     "reference_contexts" (texts of the chunks that SHOULD be retrieved),
     "answerable" (false for trap questions)}

Ground truth comes from the same table/prose structure the ingestion pipeline
indexed, so context precision/recall can be computed non-LLM (string match
against reference_contexts). Regenerate whenever the corpus changes:

    python eval/build_golden.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_ROW_RE = re.compile(r"^\|(.+)\|\s*$")


def _table_rows(markdown: str) -> list[list[str]]:
    rows = []
    for line in markdown.splitlines():
        m = _ROW_RE.match(line.strip())
        if not m:
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if cells and not set("".join(cells)) <= {"-", ":", " "}:
            rows.append(cells)
    return rows[1:]  # drop header


def main() -> None:
    from retrieval.vector_store import QdrantStore

    store = QdrantStore()
    points, _ = store.client.scroll(collection_name=store.collection,
                                    limit=1000, with_payload=True)
    active = [p.payload for p in points if p.payload.get("status") == "active"]
    tables = [c for c in active if c.get("chunk_type") == "table"]
    children = {c.get("section_title", ""): c for c in active
                if c.get("chunk_type") == "child"}

    samples: list[dict] = []

    # ── Table lookups (6): known-answer rows spread across the schedule ─────
    table = tables[0]
    rows = _table_rows(table["text"])
    picks = [rows[i] for i in (2, 10, 23, 30, 38, len(rows) - 1) if i < len(rows)]
    for code, desc, benefit, copay, auth in (r[:5] for r in picks):
        samples.append({
            "question": f"What is the copay for {desc.lower()}?",
            "reference": f"The member copay for {desc.lower()} ({code}) is {copay}.",
            "reference_contexts": [table["text"]],
            "answerable": True,
        })

    # ── Prose questions (3) ─────────────────────────────────────────────────
    prose_qs = [
        ("4. Appeals",
         "How long does a member have to appeal a denied claim?",
         "A member or provider may appeal an adverse determination within 60 "
         "days of the denial notice."),
        ("3.2 Coordination of Benefits",
         "How are claims handled when a member has coverage with another insurer?",
         "Acme Health pays secondary to the primary plan, and combined "
         "reimbursement must not exceed the maximum benefit for the billed "
         "procedure."),
        ("2.2 Exclusions",
         "Are cosmetic procedures covered?",
         "Cosmetic procedures without reconstructive indication are excluded "
         "from outpatient coverage."),
    ]
    for section, question, reference in prose_qs:
        chunk = children.get(section)
        if not chunk:
            continue
        samples.append({
            "question": question,
            "reference": reference,
            "reference_contexts": [chunk["text"]],
            "answerable": True,
        })

    # ── Trap: not answerable from the corpus ────────────────────────────────
    samples.append({
        "question": "What is the annual out-of-pocket maximum for a family plan?",
        "reference": "The policy does not specify an annual out-of-pocket maximum.",
        "reference_contexts": [],
        "answerable": False,
    })

    out = Path(__file__).parent / "golden.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote {len(samples)} golden samples to {out}")


if __name__ == "__main__":
    main()
