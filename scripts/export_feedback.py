"""Self-healing flywheel, part 2: feedback → fine-tune hard cases.

Collects 👎-rated answers (from the UI feedback store) and reviewer-EDITED
answers (from HITL checkpoints referenced in feedback threads) into
finetuning/data/hard_cases.jsonl:

  * edited answers become ready training examples (reviewer text = target)
  * plain 👎 rows are emitted with an empty target for manual curation

Merge curated hard cases into train.jsonl before the next QLoRA run — bad
answers literally become the next model's training data.

    python scripts/export_feedback.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FEEDBACK_DB = Path("data/feedback.db")
OUT = Path("finetuning/data/hard_cases.jsonl")


def main() -> None:
    if not FEEDBACK_DB.exists():
        raise SystemExit("no feedback captured yet (data/feedback.db missing)")
    from agent.graph import build_graph
    from finetuning.build_dataset import SYSTEM_PROMPT  # same trained persona

    graph = build_graph()
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM feedback WHERE rating = -1 ORDER BY created"
    ).fetchall()

    cases = []
    for row in rows:
        state = graph.get_state(
            {"configurable": {"thread_id": row["thread_id"]}}
        ).values or {}
        target = ""
        # A reviewer-edited answer is a ground-truth correction — use it.
        if state.get("reviewer_verdict") == "edited" and state.get("final_answer"):
            target = state["final_answer"]
        cases.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["query"] or state.get("query", "")},
                {"role": "assistant", "content": target},  # empty = curate manually
            ],
            "_meta": {
                "thread_id": row["thread_id"],
                "bad_answer": row["answer"],
                "comment": row["comment"],
                "grounding_score": row["grounding_score"],
                "judge_score": row["judge_score"],
                "needs_curation": not target,
            },
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        for case in cases:
            fh.write(json.dumps(case, ensure_ascii=False) + "\n")
    ready = sum(1 for c in cases if not c["_meta"]["needs_curation"])
    print(f"Wrote {len(cases)} hard cases to {OUT} "
          f"({ready} ready, {len(cases) - ready} need manual target answers)")


if __name__ == "__main__":
    main()
