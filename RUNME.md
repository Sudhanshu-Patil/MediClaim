# RUNME — Phase 1: Infrastructure + Document Ingestion Pipeline

This phase delivers the Docker stack (Qdrant, Neo4j, Redis, Langfuse) and the
async ingestion pipeline: Docling parse → version check → hierarchical chunk →
metadata → cross-reference edges → L3-cached embedding → Qdrant + Neo4j upsert.

## Prerequisites

- Docker Desktop (running)
- Python 3.10–3.12
- ~3 GB free disk (Docling's TableFormer/layout models + the bge-base
  embedding model download automatically on first run — internet needed once)

## 1. Bring up the stack

```bash
cp .env.example .env        # then change the "change-me" values
docker compose up -d
docker compose ps           # wait until every service is "healthy"
```

| Service  | URL                     | Notes                                        |
|----------|-------------------------|----------------------------------------------|
| Qdrant   | http://localhost:6333/dashboard | vector store                          |
| Neo4j    | http://localhost:7474   | login `neo4j` / your `NEO4J_PASSWORD`        |
| Redis    | localhost:6379          | L3 embedding cache + Celery broker           |
| Langfuse | http://localhost:3000   | create an account + project on first visit; put the project keys in `.env` (used from Phase 4) |

> Langfuse runs as **v2** (single container + Postgres). v3 additionally
> requires ClickHouse/MinIO/worker containers; swap in later if needed.

## 2. Install Python dependencies

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

## 3. Generate the sample test PDF

A 5-page policy document with numbered sections, "see Section X.X"
cross-references, and a 45-row fixed-column reimbursement table that spans
multiple pages:

```bash
python scripts/generate_sample_pdf.py
```

## 4. Run a test ingestion

**Quick path (no worker, pipeline runs in-process):**

```bash
python scripts/ingest.py sample_docs/sample_policy.pdf --source-type policy --sync
```

**Async path (production-shaped, via Celery):**

```bash
# terminal 1 — worker (from the repo root; --pool=solo is required on Windows)
celery -A ingestion.tasks worker --loglevel=info --pool=solo

# terminal 2 — enqueue
python scripts/ingest.py sample_docs/sample_policy.pdf --source-type policy
```

First run is slow (model downloads + TableFormer inference); repeat runs are
fast. The result JSON reports chunk counts, table chunks, REFERENCES edges,
and L3 embedding-cache hits/misses.

## 5. Verify

**Qdrant** — active chunks with payloads:

```bash
curl http://localhost:6333/collections/medclaim_chunks
curl -X POST http://localhost:6333/collections/medclaim_chunks/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"filter":{"must":[{"key":"chunk_type","match":{"value":"table"}}]},"limit":3,"with_payload":true}'
```

**Neo4j** — open http://localhost:7474 and run:

```cypher
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) RETURN d.file_name, d.version, count(c);
MATCH (a:Chunk)-[r:REFERENCES]->(b:Chunk) RETURN a.section_title, r.raw_text, b.section_title;
MATCH (s:Chunk)-[:SUMMARIZES]->(t:Chunk) RETURN s.text, t.section_title;
```

**Redis L3 cache** — embedding entries present:

```bash
docker exec medclaim-redis redis-cli --scan --pattern "l3:emb:*" | head
```

## 6. Exercise the freshness / supersede logic

```bash
# 1. Re-ingest the identical file → action: "skipped_unchanged", zero work done
python scripts/ingest.py sample_docs/sample_policy.pdf --sync

# 2. Generate version 2 (changed MRI benefit, one extra table row) and ingest
python scripts/generate_sample_pdf.py --version 2
python scripts/ingest.py sample_docs/sample_policy.pdf --sync
```

The second ingest reports `action: "changed"`, `doc_version: 2`, and
`superseded: <doc_id>:v1`. Verify in Neo4j:

```cypher
MATCH (new:Document)-[:SUPERSEDES]->(old:Document)
RETURN new.uid, new.status, old.uid, old.status;   // old is status: superseded
```

and in Qdrant (old chunks kept for audit, excluded by the default filter):

```bash
curl -X POST http://localhost:6333/collections/medclaim_chunks/points/count \
  -H "Content-Type: application/json" \
  -d '{"filter":{"must":[{"key":"status","match":{"value":"superseded"}}]},"exact":true}'
```

Unchanged prose between v1 and v2 also shows up as `embedding_cache_hits > 0`
— the Redis L3 cache skipping re-embedding (README §6).

## 7. Query the index (roadmap step 3 — hybrid retrieval + reranking)

```bash
python scripts/query.py "What is the copay for an MRI of the brain?"
python scripts/query.py "How do I appeal a denial?" --top-k 5 --json
python scripts/query.py "..." --no-graph      # vector-only ablation
python scripts/query.py "..." --no-rerank     # RRF order, no cross-encoder
```

Pipeline per query: embed (bge, L3-cached) → Qdrant search + Neo4j
fulltext/edge-expansion search in parallel branches (both filtered to
`status = active`) → **RRF fusion** (k=60) → table summaries swapped for
their full atomic tables → **cross-encoder reranking** → top-k chunks with
doc/section/page/bbox citation metadata.

Reranker notes: the repo default is `BAAI/bge-reranker-base` (~1 GB RAM). On
low-memory machines set `RERANKER_MODEL=Xenova/ms-marco-MiniLM-L-6-v2`
(~80 MB) or `RERANKER_ENABLED=0`. If the model fails to load, retrieval
degrades to fused order and marks results `reranked=false` — loudly, never
silently.

## 8. Fine-tuning (roadmap step 1 — QLoRA on free Colab T4)

Trains a LoRA adapter on Llama 3.2 3B Instruct, merges it, converts to GGUF,
and serves it locally via Ollama. Training runs on Colab (GPU); only the
dataset build runs on your laptop.

```bash
# 1. Build the instruction dataset from the ingested policy chunks + MedQuad
#    (stack must be up so the citation examples use real chunk_ids):
python finetuning/build_dataset.py --medquad 2000
#    -> finetuning/data/{train,val}.jsonl  (commit these)

# 2. Train + convert on Colab:
#    open notebooks/colab_qlora.ipynb, set runtime to T4 GPU, run all cells.
#    Requires HF access to meta-llama/Llama-3.2-3B-Instruct (gated, free) and
#    an HF write token. Pushes the q4_K_M GGUF to your HF Hub repo.

# 3. Deploy locally:
huggingface-cli download <you>/medclaim-llama3.2-3b-gguf \
    medclaim-llama3.2-3b-q4_K_M.gguf --local-dir finetuning/models
ollama create medclaim-llm -f finetuning/Modelfile
ollama run medclaim-llm
```

**Dataset composition** — two families the model must tell apart: RAG turns
with `CONTEXT` blocks train JSON `{"answer", "citations":[chunk_id...]}`
output (including refusals when context can't answer); context-free MedQuad
turns train plain-text medical answers. The citation-example count scales with
how many documents you've ingested — with only the one sample policy it's
~150, so keep `--medquad` modest (e.g. 300–500) to avoid drowning the citation
behavior, or ingest more documents first.

## 9. Ask the agent (roadmap step 4 — LangGraph)

Needs the fine-tuned model served via Ollama (§8), or any Ollama endpoint in
`LLM_BASE_URL`/`LLM_MODEL`.

```bash
python scripts/ask.py "What is the copay for an MRI of the brain?"
```

Graph: router (vector/graph/hybrid heuristic) → hybrid retrieval (§7) →
schema-constrained generation (JSON answer + chunk_id citations, Ollama JSON
mode) → citation validation → LLM-as-judge → risk gate. Failed validation,
judge score < `JUDGE_THRESHOLD`, or a high-risk query pattern pauses the run
with a LangGraph `interrupt()` and prints a review packet:

```bash
python scripts/ask.py --resume <thread_id> --verdict approved
python scripts/ask.py --resume <thread_id> --verdict edited --answer "..." --note "..."
python scripts/ask.py --resume <thread_id> --verdict rejected --note "..."
```

Checkpointing uses Redis when `langgraph-checkpoint-redis` is installed
(paused reviews survive restarts); otherwise an in-memory saver (dev only —
resume must happen in the same process). The LLM client has a circuit
breaker: 3 consecutive failures open the circuit for 30 s (fail fast instead
of stacking timeouts, README §12).

## Troubleshooting

- **`std::bad_alloc` during parsing, segfaults, or `OSError 1455` ("paging
  file is too small")**: Docling's layout + TableFormer models need roughly
  2 GB of free memory during parsing. On an 8 GB machine, close other apps
  and stop unrelated containers, or enlarge the Windows page file (System →
  Advanced → Performance → Virtual memory). The parser also auto-falls back
  to the lighter pypdfium2 backend when the default docling-parse backend
  fails (a known Docling issue,
  https://github.com/docling-project/docling/issues/3671). OCR is disabled by
  default (`DOCLING_DO_OCR=0`) to save several hundred MB — turn it on for
  scanned documents.
- **`bad allocation` from onnxruntime while loading the embedding model**:
  same memory pressure. Switch `.env` to `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5`
  and `EMBEDDING_DIM=384` (delete the Qdrant collection first if it was
  created at 768: `curl -X DELETE http://localhost:6333/collections/medclaim_chunks`).
- **Segfault when loading TableFormer right after an earlier crashed run**:
  a partially-downloaded model cache. Delete
  `%USERPROFILE%\.cache\huggingface\hub\models--docling-project--docling-models`
  and rerun.
- **`ollama run` fails with `cudaMalloc failed: out of memory`**: the GPU
  lacks ~2 GB free VRAM for the model. Either free VRAM (close browsers/apps
  using the GPU) or create the CPU variant instead:
  `ollama create medclaim-llm -f finetuning/Modelfile.cpu` (runs in system
  RAM; `num_gpu 0`). Partial offload: edit `num_gpu` to the layer count that
  fits.
- **Celery on Windows** hangs without `--pool=solo`.
- **`neo4j` unhealthy at first**: it takes ~30–40 s to boot; the healthcheck
  allows for this. Check `docker compose logs neo4j`.
- **tiktoken offline**: the chunker falls back to a word-count token estimate
  automatically; behavior is otherwise identical.
- **Port collisions**: every published port is overridable in `.env`
  (`QDRANT_HTTP_PORT`, `NEO4J_HTTP_PORT`, `LANGFUSE_PORT`, `REDIS_PORT`, ...).
  A telltale Redis symptom: `AuthenticationError: HELLO must be called ...`
  usually means another (password-protected) Redis from a different project
  already owns port 6379 — set `REDIS_PORT=6380` and update `REDIS_URL` /
  `CELERY_*` in `.env` to match. Also check `docker port medclaim-redis`
  actually shows a host binding: a container first created while the port was
  taken can end up running with no published port until you
  `docker compose up -d --force-recreate redis`.
