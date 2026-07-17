"""Ask the MedClaim agent a question (roadmap step 4 CLI).

    python scripts/ask.py "What is the copay for an MRI of the brain?"

If the answer is gated for human review the command prints the review packet
and exits; resume as the reviewer with:

    python scripts/ask.py --resume <thread_id> --verdict approved
    python scripts/ask.py --resume <thread_id> --verdict edited \
        --answer "corrected text" --note "fixed amount"
    python scripts/ask.py --resume <thread_id> --verdict rejected --note "..."

Requires the Docker stack up and an Ollama-compatible LLM endpoint
(LLM_BASE_URL / LLM_MODEL in .env; default http://localhost:11434 / medclaim-llm).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def print_result(state: dict) -> None:
    print(f"\nSTATUS: {state.get('status')}")
    print(f"ANSWER: {state.get('final_answer')}")
    citations = state.get("final_citations") or []
    if citations:
        by_id = {c["chunk_id"]: c for c in state.get("chunks", [])}
        print("CITATIONS:")
        for cid in citations:
            c = by_id.get(cid, {})
            loc = f"p.{c.get('page_number')}" if c.get("page_number") is not None else ""
            print(f"  - {cid}  ({c.get('doc_name')} v{c.get('doc_version')} — "
                  f"{c.get('section_title')} {loc})")
    if state.get("judge_score") is not None:
        print(f"JUDGE: {state.get('judge_score')} ({state.get('judge_reason','')})")
    if state.get("grounding_checked"):
        print(f"GROUNDING: {state.get('grounding_score')} "
              f"({len(state.get('ungrounded_sentences', []))} ungrounded sentence(s))")
    if state.get("pii_redacted"):
        print(f"PII REDACTED: {', '.join(state['pii_redacted'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help="the question to ask")
    parser.add_argument("--source-type", default=None,
                        choices=["policy", "clinical_guideline", "claim_note"])
    parser.add_argument("--thread", default=None, help="thread id (default: random)")
    parser.add_argument("--resume", default=None, metavar="THREAD_ID",
                        help="resume a paused review instead of asking")
    parser.add_argument("--verdict", default=None,
                        choices=["approved", "edited", "rejected"])
    parser.add_argument("--answer", default=None, help="replacement answer (verdict=edited)")
    parser.add_argument("--note", default=None, help="reviewer note")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from langgraph.types import Command

    from agent.graph import build_graph, run_agent

    app = build_graph()

    if args.resume:
        if not args.verdict:
            raise SystemExit("--resume requires --verdict")
        config = {"configurable": {"thread_id": args.resume}}
        payload = {"verdict": args.verdict, "note": args.note, "answer": args.answer}
        state = run_agent(app, Command(resume=payload), config)
    else:
        if not args.query:
            raise SystemExit("provide a query (or --resume THREAD_ID)")
        thread_id = args.thread or uuid.uuid4().hex[:12]
        config = {"configurable": {"thread_id": thread_id}}
        state = run_agent(
            app, {"query": args.query, "source_type": args.source_type}, config
        )
        if "__interrupt__" in state:
            packet = state["__interrupt__"][0].value
            print("\n*** PAUSED FOR HUMAN REVIEW ***")
            print(json.dumps(packet, indent=2))
            print(f"\nResume with:\n  python scripts/ask.py --resume {thread_id} "
                  f"--verdict approved|edited|rejected [--answer ...] [--note ...]")
            return

    if args.json:
        keep = ("status", "final_answer", "final_citations", "judge_score",
                "judge_reason", "reviewer_verdict", "route", "grounding_score",
                "ungrounded_sentences", "pii_redacted", "input_block_reason")
        print(json.dumps({k: state.get(k) for k in keep}, indent=2))
    else:
        print_result(state)


if __name__ == "__main__":
    main()
