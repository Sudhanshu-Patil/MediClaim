"""Roadmap step 9: ingestion load test + freshness verification at scale.

Generates a synthetic claim-note corpus and pushes it through the REAL
pipeline components (chunker → dense+sparse embedding with the Redis L3
cache → Qdrant upsert → Neo4j version registry), then measures:

  * ingestion throughput (docs/min, chunks/s) and the L3 cache's effect on
    an idempotent re-run
  * retrieval latency (p50/p95 over mixed queries) at small vs loaded corpus
  * freshness at scale: re-ingest a changed subset, verify supersede counts
  * Qdrant footprint

Honest scope note: Docling parsing is EXCLUDED here (claim notes are built
as pre-parsed documents). Parse cost is reported separately from measured
single-doc runs (~10–20 s/doc CPU on this machine, embarrassingly parallel
across Celery workers) — at 100k docs, parsing dominates and is a worker-
count problem, not an architecture problem. Everything downstream (embed,
upsert, retrieve, supersede) is measured for real below.

    python scripts/load_test.py --docs 1500
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FIRST_NAMES = ["A. Sharma", "R. Patel", "S. Kim", "J. Okafor", "M. Garcia",
               "L. Chen", "T. Nguyen", "K. Ivanov", "P. Rossi", "D. Yamada"]
PROCEDURES = [("OP-1001", "general practitioner consultation"),
              ("OP-1003", "telehealth consultation"),
              ("OP-3003", "MRI brain without contrast"),
              ("OP-4001", "physical therapy session"),
              ("OP-5004", "screening colonoscopy"),
              ("OP-8001", "mental health counselling"),
              ("OP-9001", "dietitian consultation"),
              ("OP-9004", "home-based sleep study")]
FINDINGS = [
    "Member reports persistent symptoms; provider recommends follow-up in 4 weeks.",
    "Documentation includes referral letter and diagnostic imaging order.",
    "Provider is in-network; eligibility confirmed on date of service.",
    "Claim includes coordination-of-benefits information from the primary insurer.",
    "Prior authorization reference attached per the reimbursement schedule.",
    "Billed amount exceeds the scheduled maximum benefit; adjustment applied.",
    "Diagnosis code supports medical necessity for the billed procedure.",
    "Member requested an itemized statement; provided by the billing office.",
]


def synth_claim_note(i: int, rng: random.Random, revision: int = 1):
    """Build a pre-parsed claim note (2–4 paragraph blocks + heading)."""
    from ingestion.parser import ParsedBlock, ParsedDocument, SourceFormat

    code, desc = rng.choice(PROCEDURES)
    provider = rng.choice(FIRST_NAMES)
    amount = round(rng.uniform(30, 900), 2)
    day = rng.randint(1, 28)
    paragraphs = [
        f"Claim note for claim CLM-{20000 + i}. Service: {desc} ({code}) rendered "
        f"on 2026-{rng.randint(1, 6):02d}-{day:02d} by provider {provider}. "
        f"Billed amount {amount:.2f} USD."
        + (f" REVISION {revision}: amount corrected after audit." if revision > 1 else ""),
        rng.choice(FINDINGS) + " " + rng.choice(FINDINGS),
    ]
    if rng.random() < 0.5:
        paragraphs.append(rng.choice(FINDINGS))
    blocks = [ParsedBlock(kind="heading", text=f"Claim note CLM-{20000 + i}",
                          heading_level=1, paragraph_index=0)]
    blocks += [ParsedBlock(kind="paragraph", text=p, paragraph_index=j + 1)
               for j, p in enumerate(paragraphs)]
    return ParsedDocument(source_path=f"synthetic://claim_{i}",
                          file_name=f"claim_note_{i:05d}.docx",
                          source_format=SourceFormat.DOCX, elements=blocks)


def ingest_batch(n_docs: int, seed: int, revision: int = 1,
                 subset: list[int] | None = None) -> dict:
    """Chunk + embed (L3-cached) + upsert a batch through real components."""
    import hashlib

    from config import get_settings
    from ingestion.chunker import chunk_document
    from ingestion.embedder import CachedEmbedder
    from ingestion.freshness import compute_doc_id
    from ingestion.metadata_schema import SourceType
    from retrieval.graph_store import Neo4jStore, document_uid
    from retrieval.vector_store import QdrantStore

    settings = get_settings()
    vectors_store = QdrantStore()
    vectors_store.ensure_collection()
    graph = Neo4jStore()
    graph.ensure_constraints()
    embedder = CachedEmbedder()
    rng = random.Random(seed)

    doc_ids = subset if subset is not None else list(range(n_docs))
    t0 = time.perf_counter()
    total_chunks = cache_hits = cache_misses = superseded_docs = 0
    pending_texts, pending_hashes, pending_ids, pending_payloads = [], [], [], []

    def flush():
        nonlocal total_chunks, cache_hits, cache_misses
        if not pending_ids:
            return
        dense = embedder.embed(pending_texts, pending_hashes)
        cache_hits += embedder.last_cache_hits
        cache_misses += embedder.last_cache_misses
        sparse = embedder.embed_sparse(pending_texts)
        vectors_store.upsert_chunks(pending_ids, dense, pending_payloads,
                                    sparse_vectors=sparse,
                                    batch_size=settings.embed_batch_size)
        total_chunks += len(pending_ids)
        pending_texts.clear(); pending_hashes.clear()
        pending_ids.clear(); pending_payloads.clear()

    try:
        for i in doc_ids:
            parsed = synth_claim_note(i, random.Random(seed * 100003 + i), revision)
            doc_id = compute_doc_id(parsed.file_name, SourceType.CLAIM_NOTE)
            content_hash = hashlib.sha256(
                "".join(b.text for b in parsed.elements).encode()).hexdigest()
            latest = graph.get_latest_document(doc_id, status="active")
            if latest and latest.get("content_hash") == content_hash:
                continue  # UNCHANGED skip (idempotent re-run path)
            version = (int(latest["version"]) + 1) if latest else 1
            chunks = chunk_document(parsed, doc_id=doc_id, doc_version=version,
                                    source_type=SourceType.CLAIM_NOTE,
                                    doc_hash=content_hash)
            for chunk in chunks:
                payload = chunk.metadata.to_qdrant_payload()
                payload["text"] = chunk.text
                pending_texts.append(chunk.text)
                pending_hashes.append(chunk.metadata.chunk_hash)
                pending_ids.append(chunk.chunk_id)
                pending_payloads.append(payload)
            if len(pending_ids) >= settings.embed_batch_size * 2:
                flush()
            graph.upsert_document({
                "uid": document_uid(doc_id, version), "doc_id": doc_id,
                "version": version, "content_hash": content_hash,
                "file_name": parsed.file_name,
                "source_type": "claim_note", "status": "active",
            })
            if latest:  # CHANGED: supersede previous version
                vectors_store.mark_superseded(doc_id, int(latest["version"]),
                                              document_uid(doc_id, version))
                graph.mark_document_superseded(latest["uid"],
                                               document_uid(doc_id, version))
                graph.create_supersedes_edge(document_uid(doc_id, version),
                                             latest["uid"])
                superseded_docs += 1
        flush()
    finally:
        graph.close()

    elapsed = time.perf_counter() - t0
    return {"docs": len(doc_ids), "chunks": total_chunks,
            "elapsed_s": round(elapsed, 1),
            "docs_per_min": round(len(doc_ids) / elapsed * 60, 1),
            "chunks_per_s": round(total_chunks / elapsed, 1) if elapsed else 0,
            "l3_cache_hits": cache_hits, "l3_cache_misses": cache_misses,
            "superseded_docs": superseded_docs}


QUERIES = [
    "What is the copay for an MRI of the brain?",
    "claim CLM-20042 billed amount",
    "sleep study claims by provider S. Kim",
    "How long does a member have to appeal a denied claim?",
    "OP-5004 screening colonoscopy prior authorization",
    "telehealth consultation claims coordination of benefits",
    "dietitian consultation maximum benefit",
    "claims exceeding scheduled maximum benefit",
    "Are cosmetic procedures covered?",
    "physical therapy documentation referral letter",
]


def bench_retrieval(label: str, n_iter: int = 2) -> dict:
    from retrieval.retriever import HybridRetriever

    retriever = HybridRetriever()
    try:
        retriever.retrieve("warmup query", top_k=4)  # load models outside timing
        latencies = []
        for _ in range(n_iter):
            for q in QUERIES:
                t0 = time.perf_counter()
                retriever.retrieve(q, top_k=6)
                latencies.append((time.perf_counter() - t0) * 1000)
    finally:
        retriever.close()
    latencies.sort()
    return {"label": label, "n": len(latencies),
            "p50_ms": round(statistics.median(latencies)),
            "p95_ms": round(latencies[int(len(latencies) * 0.95) - 1]),
            "max_ms": round(latencies[-1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    from retrieval.vector_store import QdrantStore

    report: dict = {"docs_requested": args.docs}

    print("== retrieval baseline (small corpus) ==")
    report["retrieval_baseline"] = bench_retrieval("baseline")
    print(report["retrieval_baseline"])

    print(f"\n== ingest {args.docs} synthetic claim notes ==")
    report["ingest_initial"] = ingest_batch(args.docs, args.seed)
    print(report["ingest_initial"])

    print("\n== idempotent re-run (all UNCHANGED, should be fast) ==")
    report["ingest_rerun"] = ingest_batch(args.docs, args.seed)
    print(report["ingest_rerun"])

    print("\n== freshness at scale: revise a 5% subset ==")
    rng = random.Random(args.seed + 1)
    subset = rng.sample(range(args.docs), max(1, args.docs // 20))
    report["ingest_revision"] = ingest_batch(args.docs, args.seed, revision=2,
                                             subset=subset)
    print(report["ingest_revision"])

    store = QdrantStore()
    report["qdrant"] = {
        "total_points": store.count(),
        "active": store.count(status="active"),
        "superseded": store.count(status="superseded"),
    }
    print("\nqdrant:", report["qdrant"])
    assert report["ingest_revision"]["superseded_docs"] == len(subset), \
        "supersede count mismatch!"

    print("\n== retrieval at loaded corpus ==")
    report["retrieval_loaded"] = bench_retrieval("loaded")
    print(report["retrieval_loaded"])

    out = Path("eval/results/load_test.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
