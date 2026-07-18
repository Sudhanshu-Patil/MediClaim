"""Central environment-driven settings for the MedClaim ingestion pipeline.

Everything is read from the environment (with .env support) so the same code
runs on a laptop, in Docker, or in CI without edits. Import via:

    from config import get_settings
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

# Windows: huggingface_hub's default symlink cache needs Developer Mode /
# admin rights (WinError 1314 otherwise). Copying instead costs some disk but
# always works — model downloads (Docling TableFormer, bge-base) depend on it.
if os.name == "nt":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # Qdrant (qdrant_api_key: required for Qdrant Cloud, unused/None for local Docker)
    qdrant_url: str = field(default_factory=lambda: _env("QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY") or None)
    # QdrantClient's own default is untimed/very short — fine for local
    # Docker's sub-millisecond round trips, but a real network hop to a
    # free-tier cloud cluster (esp. a cold collection's first write) can
    # legitimately take longer. Measured: local Docker never needed this;
    # Qdrant Cloud threw WriteTimeout on the default.
    qdrant_timeout_s: int = field(default_factory=lambda: _env_int("QDRANT_TIMEOUT_S", 30))
    qdrant_collection: str = field(default_factory=lambda: _env("QDRANT_COLLECTION", "medclaim_chunks"))

    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: _env("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: _env("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: _env("NEO4J_PASSWORD", "medclaim-local-dev"))

    # Redis / Celery
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    celery_broker_url: str = field(default_factory=lambda: _env("CELERY_BROKER_URL", "redis://localhost:6379/1"))
    celery_result_backend: str = field(default_factory=lambda: _env("CELERY_RESULT_BACKEND", "redis://localhost:6379/2"))
    l3_embedding_cache_ttl: int = field(default_factory=lambda: _env_int("L3_EMBEDDING_CACHE_TTL", 0))

    # Embeddings
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"))
    embedding_dim: int = field(default_factory=lambda: _env_int("EMBEDDING_DIM", 768))
    embed_batch_size: int = field(default_factory=lambda: _env_int("EMBED_BATCH_SIZE", 96))

    # Chunking (README §5: 512–1024 tokens, 10–20% overlap)
    chunk_min_tokens: int = field(default_factory=lambda: _env_int("CHUNK_MIN_TOKENS", 512))
    chunk_max_tokens: int = field(default_factory=lambda: _env_int("CHUNK_MAX_TOKENS", 1024))
    chunk_target_tokens: int = field(default_factory=lambda: _env_int("CHUNK_TARGET_TOKENS", 768))
    chunk_overlap_tokens: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP_TOKENS", 100))
    parent_max_tokens: int = field(default_factory=lambda: _env_int("PARENT_MAX_TOKENS", 3072))

    # Retrieval + reranking (roadmap step 3)
    reranker_model: str = field(default_factory=lambda: _env("RERANKER_MODEL", "BAAI/bge-reranker-base"))
    reranker_enabled: bool = field(default_factory=lambda: _env("RERANKER_ENABLED", "1") == "1")
    # 6, not 8: measured on the current fine-tune — at 8 context chunks the
    # model flips into degenerate question-babble (and breaks its JSON
    # format); at 6 it answers correctly. Another training-distribution
    # ceiling (dataset examples carried 1-2 context blocks).
    retrieval_top_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_TOP_K", 6))
    retrieval_vector_top_n: int = field(default_factory=lambda: _env_int("RETRIEVAL_VECTOR_TOP_N", 20))
    retrieval_graph_top_n: int = field(default_factory=lambda: _env_int("RETRIEVAL_GRAPH_TOP_N", 20))
    retrieval_rrf_k: int = field(default_factory=lambda: _env_int("RETRIEVAL_RRF_K", 60))
    retrieval_sparse_weight: float = field(default_factory=lambda: float(_env("RETRIEVAL_SPARSE_WEIGHT", "0.5")))
    # Retrieve small, read big: match/rerank on children, feed the generator
    # their full parent sections. DEFAULT OFF — measured with the current
    # fine-tune: parent-style contexts are out-of-distribution for a model
    # trained on child/table contexts (trap refusal 1.0 -> 0.0, one context
    # echo). Enable after a v2 fine-tune whose data includes parent contexts.
    retrieval_expand_parents: bool = field(default_factory=lambda: _env("RETRIEVAL_EXPAND_PARENTS", "0") == "1")
    # 16, not fewer: trimming to 12 was measured to cost 0.17 context
    # precision on the golden set for ~400 ms — not a good trade. The
    # cross-encoder pass is the retrieval latency floor (~100 ms/candidate).
    retrieval_rerank_candidates: int = field(default_factory=lambda: _env_int("RETRIEVAL_RERANK_CANDIDATES", 16))

    # Grounding / NLI entailment check (roadmap step 5, README §8)
    grounding_model: str = field(default_factory=lambda: _env("GROUNDING_MODEL", "cross-encoder/nli-deberta-v3-xsmall"))
    grounding_enabled: bool = field(default_factory=lambda: _env("GROUNDING_ENABLED", "1") == "1")
    # Per-sentence entailment probability below this = ungrounded sentence.
    entailment_threshold: float = field(default_factory=lambda: float(_env("ENTAILMENT_THRESHOLD", "0.5")))
    # Grounded fraction below this routes the answer to human review.
    grounding_review_threshold: float = field(default_factory=lambda: float(_env("GROUNDING_REVIEW_THRESHOLD", "0.8")))

    # GraphRAG scoping (README §3)
    graph_scoped_source_types: frozenset[str] = field(
        default_factory=lambda: frozenset(
            s.strip()
            for s in _env("GRAPH_SCOPED_SOURCE_TYPES", "policy,clinical_guideline").split(",")
            if s.strip()
        )
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
