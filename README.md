# MedClaim-Agentic-RAG

**A zero-cost, production-shaped agentic RAG system for clinical & insurance claims adjudication** — combining QLoRA fine-tuning, local deployment, hybrid GraphRAG + vector retrieval, LangGraph orchestration, human-in-the-loop review, and full observability/evaluation.

Built to demonstrate end-to-end LLM engineering: fine-tuning → agentic RAG → production hardening (scale, freshness, grounding, citations) — not just a chatbot demo.

---

## 1. Problem Statement / Use Case

Insurance claims adjudicators and healthcare providers need to query medical policies, clinical guidelines, and claim histories, cross-reference diagnosis–drug–procedure relationships to flag anomalies, and receive claim decisions with **traceable, citable justification** — not an ungrounded paragraph. The system must:

- Scale to ~100k documents without degrading
- Never answer from a stale/superseded document once a newer version is uploaded
- Ground every generated claim in a specific, citable source chunk
- Escalate high-risk or low-confidence decisions to a human reviewer

This spans two domains deliberately: **healthcare** (clinical reasoning) and **BFSI** (insurance adjudication).

---

## 2. Architecture Diagram

See [`architecture.mermaid`](./architecture.mermaid) for the full diagram. High-level flow:

```
Document Upload → Docling Parse → Version Check → Hierarchical Chunk + Metadata
   → Embed (Redis-cached) → Qdrant (vector) + Neo4j (GraphRAG, scoped subset)

User Query → Guardrails (in) → Redis Cache Check → LangGraph Router
   → [Vector Retriever ∥ Graph Retriever] → Reranker → Fine-tuned LLM (+ MCP tools)
   → Grounding Check → LLM-as-Judge → [HITL if high-risk] → Guardrails (out)
   → Cited, Highlighted Response → Cache

Cross-cutting: LangSmith + Langfuse tracing, RAGAS offline eval, MLflow experiment tracking
```

---

## 3. Key Architectural Decisions

**GraphRAG vs. LangGraph — not competitors, different layers.**
GraphRAG is a *retrieval* technique (knowledge-graph traversal for entity/relationship reasoning). LangGraph is an *orchestration* framework (agent state machines, conditional routing, HITL interrupts, checkpointing). The router decides vector vs. graph vs. hybrid retrieval per query; both feed a shared reranker.

**GraphRAG is scoped, not applied everywhere.** LLM-based entity/relationship extraction over 100k documents is the actual bottleneck at scale — not the vector store. GraphRAG is deliberately scoped to relationship-dense, high-value documents (policies, clinical guidelines). High-volume, low-relationship documents (individual claim notes) go through vector-only retrieval. This hybrid scoping is a stated architectural trade-off, not a shortcut.

**Freshness is identity-based, not timestamp-guessing.** Every document has a `doc_id` + `version` + `content_hash`. New versions supersede old ones explicitly (`status: superseded`, `superseded_by`), retrieval filters exclude superseded chunks by default, and a `SUPERSEDES` graph edge preserves audit history — insurance claims can be disputed, so old answers must remain explainable, not vanish.

**Citations are structural, not post-hoc.** The generator is schema-constrained (via Guardrails) to emit the `chunk_id`s it actually used. This is what makes grounding checks, judge scoring, and line-level highlighting possible at all — provenance is captured at generation time, not reconstructed afterward.

**Cross-references resolve two ways.** Most "see section 4.2" references are pre-linked as graph edges at ingestion time (cheap, zero query-time latency). The long tail (cross-document references, or references to a doc ingested after the graph was last built) falls back to an agentic MCP tool call (`resolve_reference`), capped at 1–2 calls per response to bound latency.

---

## 4. Full Tech Stack (100% Free / Local / Open Source)

| Layer | Tool | Notes |
|---|---|---|
| Base LLM | Llama 3.1 8B / Mistral 7B / Qwen2.5 7B | open weights |
| Fine-tuning | HF PEFT + QLoRA + bitsandbytes | free Colab/Kaggle T4 GPU |
| Local deployment | Ollama (GGUF quantized) | merged LoRA adapter, zero inference cost |
| Vector DB | Qdrant (Docker, local) | HNSW indexing + quantization for scale |
| Graph DB + GraphRAG | Neo4j Community Edition + Microsoft GraphRAG (OSS) | scoped to policy/guideline docs |
| Orchestration | LangGraph | routing, HITL interrupts, checkpointing |
| Reranker | BAAI/bge-reranker-base (local) | cross-encoder re-scoring post-retrieval |
| Parsing | Docling (IBM, OSS) | PDF/DOCX/PPTX, TableFormer for complex tables |
| Chunking | LlamaIndex `HierarchicalNodeParser` / sentence-window | parent-child + atomic table chunks |
| Guardrails | Guardrails AI / NeMo Guardrails (OSS) | input PII/jailbreak, output schema/hallucination |
| Grounding | Local NLI/entailment model | runtime per-sentence entailment check |
| HITL | LangGraph native `interrupt()` | checkpointed in Redis |
| LLM-as-judge | Local fine-tuned or secondary small model | pre-response faithfulness/relevance scoring |
| Caching / Queue | Redis | 4 uses — see §6 |
| Async ingestion | Celery + Redis broker | idempotent, retryable batch jobs |
| Observability | LangSmith (free dev tier) + Langfuse (self-hosted Docker) | full trace history, no usage caps |
| Evaluation | RAGAS (OSS) | faithfulness, context precision/recall, relevance |
| MCP | Custom Python MCP server | policy-lookup, claim-DB query, fraud-flag, resolve_reference |
| API | FastAPI + slowapi (rate limiting) | wraps LangGraph app |
| UI | Streamlit / Gradio | optionally deployed on HF free CPU Space (no card needed) |
| Experiment tracking | MLflow (local) | fine-tuning run comparisons |

**Zero-cost note:** Hugging Face's free **CPU Basic** Spaces tier requires no credit card at all — only PRO/GPU tiers do. The fine-tuned model itself is served locally via Ollama regardless, so HF Spaces (if used) only hosts a thin UI, not the model.

---

## 5. Document Parsing & Chunking (Production-Grade)

**Parsing:** Docling handles PDF, DOCX, and PPTX uniformly. Its TableFormer model reconstructs actual row/column structure — including tables that span multiple pages (merged as one logical table) and complex inline fixed-column, multi-row tables with merged cells — rather than flattening them into garbled text. Parsing happens once, asynchronously, at ingestion time, never at query time, keeping retrieval latency low.

**Chunking strategy:**

| Content type | Strategy | Size | Overlap |
|---|---|---|---|
| Prose (policy text, clinical notes) | Sentence-window / recursive character | 512–1024 tokens | 10–20% (~100 tokens) |
| Tables | Atomic — never split | whole table as one chunk | plus a 1–2 line summary chunk pointing back to it |
| Sections | Parent-child hierarchy | header+section as parent, sentences as children | children inherit parent's metadata |

**Per-chunk metadata schema:**

| Field | Purpose |
|---|---|
| `chunk_id` | unique, deterministic (hash of doc_id + chunk_index) |
| `doc_id`, `doc_version` | freshness + versioning |
| `section_title`, `page_number` / `slide_index` / `paragraph_index` | human-checkable citation |
| `bbox` | PDF bounding box, for line-level highlighting |
| `source_type` | policy / clinical-guideline / claim-note |
| `effective_date`, `status` | staleness filtering (active / superseded) |
| `doc_hash` | change detection |
| `ingestion_timestamp` | audit trail |

---

## 6. Redis — Four Distinct Roles

| Role | What it does | Why it matters |
|---|---|---|
| L1 — exact-match cache | `hash(query + doc_version_state)` → answer | instant repeat-query response; auto-invalidates on new upload |
| L2 — semantic cache | query-embedding similarity → answer (GPTCache) | catches paraphrased questions |
| L3 — embedding cache | `chunk_hash` → embedding vector | skips re-embedding unchanged docs on re-ingestion — real compute savings at 100k-doc scale |
| Celery broker / LangGraph checkpointer | job queue + HITL interrupt state persistence | ingestion survives worker crashes; paused claim reviews survive process restarts |

---

## 7. Scaling to ~100k Documents

- **Async ingestion**: Celery + Redis job queue; uploads don't block on parse/embed/upsert.
- **Batched embedding + upsert**: 64–128 chunks per call — the actual difference between minutes and hours at scale.
- **Idempotent jobs**: deterministic chunk IDs, `acks_late` + exponential backoff, safe to retry without duplication.
- **GraphRAG scoping** (see §3): the real bottleneck at scale is LLM-based entity extraction, not the vector store — scope it deliberately.
- **Qdrant** handles millions of vectors on a single local instance via HNSW + scalar/binary quantization.

---

## 8. Grounding, Citations & Highlighting

1. Generator emits the answer **plus** the `chunk_id`s it used (schema-enforced via Guardrails).
2. A local NLI/entailment model checks each generated sentence against its cited chunk **at runtime** — a live guardrail, distinct from offline evaluation.
3. RAGAS runs **offline** to separately track:
   - **Context precision/recall** — is retrieval pulling the right chunks?
   - **Faithfulness** — did generation stay grounded in them?
   (Tracked separately — good retrieval + ungrounded generation is a real, common failure mode that a combined metric would hide.)
4. UI resolves each cited `chunk_id` → `doc_id + page/slide + bbox` → renders the source document with that region highlighted, **grouped by source document** when an answer draws on multiple documents.

---

## 9. Agentic Function Calling — Cross-Reference Resolution

- **Static (cheap, ingestion-time)**: cross-reference patterns ("see section 4.2") detected via regex/light NLP, stored as explicit `REFERENCES` edges in Neo4j. Resolves the common case for free, before any query happens.
- **Dynamic (agentic, query-time fallback)**: an MCP tool `resolve_reference(doc_id, section_number)` is exposed to the LangGraph agent for references that weren't pre-linked (cross-document, or a doc ingested after the graph was last built). Capped at 1–2 tool calls per response to bound latency — the agent doesn't get free rein to chase references indefinitely.

---

## 10. Human-in-the-Loop & Guardrails

- LangGraph's native `interrupt()` pauses the graph at high-risk or low-judge-score decision points; state is checkpointed in Redis so a paused review survives restarts.
- **Input guardrails**: PII redaction, jailbreak/toxicity detection.
- **Output guardrails**: schema validation (citations must resolve to real chunk_ids), hallucination flagging.

---

## 11. Observability & Evaluation

| Tool | Purpose |
|---|---|
| LangSmith | dev-time tracing/debugging of the LangGraph agent |
| Langfuse (self-hosted) | persistent trace storage, cost/latency dashboards, no usage caps |
| RAGAS | offline faithfulness, context precision/recall, answer relevance — run pre/post fine-tuning **and** pre/post adding GraphRAG, as a comparison table |
| MLflow | fine-tuning run tracking and comparison |

---

## 12. Production Hardening

- **Circuit breaker** around local LLM/reranker calls — fail fast and queue rather than cascading timeouts if Ollama is overloaded.
- **Rate limiting** at the FastAPI layer (`slowapi`).
- **Health checks / readiness probes** for Qdrant, Neo4j, Redis, and Ollama containers in Docker Compose.
- **Idempotent, retryable ingestion jobs** as covered in §7.

---

## 13. Proposed Repo Structure

```
medclaim-agentic-rag/
├── docker-compose.yml          # Qdrant, Neo4j, Redis, Langfuse
├── ingestion/
│   ├── parser.py                # Docling wrapper
│   ├── chunker.py                # hierarchical + atomic table chunking
│   ├── metadata_schema.py
│   ├── freshness.py              # version/hash checks, supersede logic
│   ├── cross_reference.py        # static REFERENCES edge detection
│   └── tasks.py                  # Celery jobs
├── retrieval/
│   ├── vector_store.py           # Qdrant client
│   ├── graph_store.py            # Neo4j / GraphRAG client
│   └── reranker.py
├── agent/
│   ├── graph.py                  # LangGraph definition
│   ├── nodes/                    # router, retrievers, generator, judge, guardrail nodes
│   └── mcp_server.py             # custom MCP tools
├── caching/
│   └── redis_client.py           # L1/L2/L3 cache logic
├── finetuning/
│   ├── qlora_train.py
│   └── merge_and_convert.py      # LoRA merge → GGUF → Ollama
├── eval/
│   ├── ragas_eval.py
│   └── mlflow_tracking.py
├── api/
│   └── main.py                   # FastAPI + slowapi
├── ui/
│   └── app.py                    # Streamlit/Gradio
├── architecture.mermaid
└── README.md
```

---

## 14. Build Roadmap

1. Fine-tune (QLoRA) → merge → convert to GGUF → serve locally via Ollama
2. Stand up Qdrant + Neo4j in Docker; build Docling → chunk → metadata → embed → upsert ingestion pipeline with Celery/Redis
3. Add reranker (bge-reranker-base)
4. Build the LangGraph agent (router, retrievers, generator, judge, guardrail nodes, `interrupt()`)
5. Wrap in Guardrails (input + output) and the grounding/entailment check
6. Instrument LangSmith + self-hosted Langfuse
7. Run RAGAS evaluation before/after fine-tuning and before/after GraphRAG
8. Build the custom MCP server + `resolve_reference` tool; deploy UI (HF free CPU Space or local + demo recording)
9. Load-test ingestion at scale (synthetic 100k-doc corpus), verify freshness/supersede logic, add production hardening (circuit breaker, rate limiting, health checks)
