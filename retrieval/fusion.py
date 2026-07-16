"""Reciprocal Rank Fusion (RRF) — merges the vector and graph result lists.

RRF is rank-based, so it needs no score normalization between Qdrant cosine
similarities and Neo4j Lucene scores (which live on incomparable scales):

    fused_score(d) = Σ_over_lists  weight_i / (k + rank_i(d))

k=60 is the standard constant from the original RRF paper (Cormack et al.,
2009); it dampens the dominance of rank-1 hits so agreement between retrievers
outweighs a single retriever's confidence.
"""

from __future__ import annotations

from typing import Optional, Sequence

DEFAULT_RRF_K = 60


def rrf_fuse(
    ranked_lists: Sequence[Sequence[str]],
    k: int = DEFAULT_RRF_K,
    weights: Optional[Sequence[float]] = None,
) -> list[tuple[str, float]]:
    """Fuse ranked ID lists into one list of (id, fused_score), best first.

    ``ranked_lists`` — each inner sequence is IDs ordered best-first from one
    retriever. ``weights`` optionally biases retrievers (default: equal).
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must match ranked_lists length")

    scores: dict[str, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + weight / (k + rank)

    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
