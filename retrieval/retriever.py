"""Hybrid retriever: vector ∥ graph → RRF fusion → cross-encoder rerank.

Implements the retrieval half of the architecture diagram's query pipeline:

    query ──embed──> Qdrant search (status=active)  ─┐
          ──text───> Neo4j fulltext + edge expansion ─┴─> RRF fuse
          → swap table summaries for their full tables
          → bge-reranker cross-encoder rescoring
          → RetrievedChunk list with citation metadata + parent context ids

Design notes:
  * Vector search targets CHILD / TABLE / TABLE_SUMMARY chunks only — parents
    aggregate their children's text, so retrieving them directly would just
    duplicate every hit at a coarser granularity. Parents come back as
    ``parent_chunk_id`` for context expansion by the generator.
  * A TABLE_SUMMARY hit is a pointer, not the payload: before reranking it is
    swapped for the full atomic table chunk it summarizes, so the reranker
    and the generator see actual rows, not the 2-line summary.
  * Every result carries doc/section/page/bbox so the later citation and
    highlighting stages (README §8) get provenance for free.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from config import get_settings
from ingestion.embedder import CachedEmbedder
from retrieval.fusion import rrf_fuse
from retrieval.graph_store import Neo4jStore
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_store import QdrantStore

logger = logging.getLogger(__name__)

_RETRIEVABLE_CHUNK_TYPES = ("child", "table", "table_summary")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float                     # rerank score, or RRF score if degraded
    fused_score: float               # RRF score before reranking
    reranked: bool
    chunk_type: str
    doc_id: str
    doc_version: int
    doc_name: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    slide_index: Optional[int] = None
    paragraph_index: Optional[int] = None
    bbox: Optional[list] = None
    parent_chunk_id: Optional[str] = None
    sources: list[str] = field(default_factory=list)  # ["vector"], ["graph"], or both


class HybridRetriever:
    def __init__(
        self,
        vector_store: Optional[QdrantStore] = None,
        graph_store: Optional[Neo4jStore] = None,
        embedder: Optional[CachedEmbedder] = None,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.settings = get_settings()
        self.vectors = vector_store or QdrantStore()
        self.graph = graph_store or Neo4jStore()
        self.embedder = embedder or CachedEmbedder()
        self.reranker = reranker or CrossEncoderReranker()

    def close(self) -> None:
        self.graph.close()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        source_type: Optional[str] = None,
        use_graph: bool = True,
        use_reranker: bool = True,
    ) -> list[RetrievedChunk]:
        s = self.settings
        top_k = top_k or s.retrieval_top_k

        # ── 1. Vector branch ────────────────────────────────────────────────
        query_vector = self.embedder.embed_query(query)
        vector_hits = self.vectors.search(
            query_vector,
            top_n=s.retrieval_vector_top_n,
            chunk_types=_RETRIEVABLE_CHUNK_TYPES,
            source_type=source_type,
        )
        payloads: dict[str, dict] = {h["chunk_id"]: h["payload"] for h in vector_hits}
        vector_ranked = [h["chunk_id"] for h in vector_hits]

        # ── 2. Graph branch (fulltext seeds + REFERENCES/SUMMARIZES/parent) ─
        graph_ranked: list[str] = []
        if use_graph:
            graph_hits = self.graph.graph_search(
                query, top_n=s.retrieval_graph_top_n, source_type=source_type
            )
            graph_ranked = [h["chunk_id"] for h in graph_hits]

        # ── 3. RRF fusion ───────────────────────────────────────────────────
        fused = rrf_fuse(
            [vector_ranked, graph_ranked] if graph_ranked else [vector_ranked],
            k=s.retrieval_rrf_k,
        )
        candidates = fused[: s.retrieval_rerank_candidates]

        # Fetch payloads for graph-only hits, then drop candidates that have
        # no payload (e.g. graph returned a parent chunk not stored… defensive)
        missing = [cid for cid, _ in candidates if cid not in payloads]
        payloads.update(self.vectors.retrieve(missing))
        candidates = [(cid, sc) for cid, sc in candidates if cid in payloads]

        # ── 4. Swap table summaries for the full tables they point at ──────
        swapped: list[tuple[str, float]] = []
        seen: set[str] = set()
        for cid, fused_score in candidates:
            payload = payloads[cid]
            # Parents (reachable via graph expansion) are context payloads,
            # not retrieval units — their children carry parent_chunk_id.
            if payload.get("chunk_type") == "parent":
                continue
            if payload.get("chunk_type") == "table_summary" and payload.get("refers_to_chunk_id"):
                target = payload["refers_to_chunk_id"]
                if target not in payloads:
                    payloads.update(self.vectors.retrieve([target]))
                if target in payloads:
                    cid = target
            if cid not in seen:
                seen.add(cid)
                swapped.append((cid, fused_score))
        candidates = swapped

        # ── 5. Cross-encoder rerank (graceful degrade to fused order) ──────
        reranked = False
        results: list[tuple[str, float, float]] = []  # (id, score, fused)
        if use_reranker:
            order = self.reranker.rerank(
                query, [payloads[cid].get("text", "") for cid, _ in candidates]
            )
        else:
            order = None
        if order is not None:
            reranked = True
            for idx, score in order[:top_k]:
                cid, fused_score = candidates[idx]
                results.append((cid, score, fused_score))
        else:
            results = [(cid, sc, sc) for cid, sc in candidates[:top_k]]

        # ── 6. Materialize with citation metadata ───────────────────────────
        graph_set, vector_set = set(graph_ranked), set(vector_ranked)
        retrieved = []
        for cid, score, fused_score in results:
            p = payloads[cid]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=p.get("text", ""),
                    score=score,
                    fused_score=fused_score,
                    reranked=reranked,
                    chunk_type=p.get("chunk_type", ""),
                    doc_id=p.get("doc_id", ""),
                    doc_version=p.get("doc_version", 0),
                    doc_name=p.get("doc_name"),
                    section_title=p.get("section_title"),
                    page_number=p.get("page_number"),
                    slide_index=p.get("slide_index"),
                    paragraph_index=p.get("paragraph_index"),
                    bbox=p.get("bbox"),
                    parent_chunk_id=p.get("parent_chunk_id"),
                    sources=[
                        s for s, present in
                        (("vector", cid in vector_set), ("graph", cid in graph_set))
                        if present
                    ],
                )
            )
        logger.info(
            "Retrieved %d chunks for %r (vector=%d, graph=%d, reranked=%s)",
            len(retrieved), query[:60], len(vector_ranked), len(graph_ranked), reranked,
        )
        return retrieved
