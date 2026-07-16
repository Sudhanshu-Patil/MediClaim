"""CLI for querying the hybrid retriever (roadmap step 3 smoke tool).

From the repo root:

    python scripts/query.py "What is the copay for an MRI of the brain?"
    python scripts/query.py "How do I appeal a denial?" --top-k 5 --json
    python scripts/query.py "..." --no-rerank --no-graph   # ablations
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--source-type", default=None,
                        choices=["policy", "clinical_guideline", "claim_note"])
    parser.add_argument("--no-graph", action="store_true", help="vector-only ablation")
    parser.add_argument("--no-rerank", action="store_true", help="skip cross-encoder")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    from retrieval.retriever import HybridRetriever

    retriever = HybridRetriever()
    try:
        results = retriever.retrieve(
            args.query,
            top_k=args.top_k,
            source_type=args.source_type,
            use_graph=not args.no_graph,
            use_reranker=not args.no_rerank,
        )
    finally:
        retriever.close()

    if args.json:
        print(json.dumps([dataclasses.asdict(r) for r in results], indent=2))
        return

    if not results:
        print("No results.")
        return

    print(f'Query: "{args.query}"  ({len(results)} results, '
          f'reranked={results[0].reranked})\n')
    for rank, r in enumerate(results, start=1):
        location = []
        if r.page_number is not None:
            location.append(f"p.{r.page_number}")
        if r.slide_index is not None:
            location.append(f"slide {r.slide_index}")
        if r.paragraph_index is not None:
            location.append(f"para {r.paragraph_index}")
        snippet = textwrap.shorten(" ".join(r.text.split()), width=220, placeholder=" …")
        print(f"#{rank}  score={r.score:.4f}  [{r.chunk_type}]  via {'+'.join(r.sources) or 'fusion'}")
        print(f"    {r.doc_name} v{r.doc_version} — {r.section_title or '(no section)'}"
              f"{'  (' + ', '.join(location) + ')' if location else ''}")
        print(f"    {snippet}")
        print(f"    chunk_id={r.chunk_id}")
        print()


if __name__ == "__main__":
    main()
