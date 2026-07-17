"""FastAPI service wrapping the LangGraph agent (roadmap step 8, README §4).

This persistent process is also the latency fix: retriever, NLI, and graph
load ONCE at startup instead of per CLI invocation (~25 s saved per query).

Endpoints:
  POST /ask            — SSE stream: token events as the answer generates,
                         then a final `result` event with citations/verdicts,
                         or a `review` event when HITL pauses the run
  POST /review/{thread}— resume a paused review (approved/edited/rejected)
  POST /upload         — ingest a PDF/DOCX/PPTX into the corpus
  GET  /source/{chunk_id}/image — the source PDF page rendered as PNG with
                         the cited region highlighted (bbox provenance)
  GET  /chunks/{chunk_id} — chunk text + metadata
  GET  /tts?text=      — free TTS audio (edge-tts) for voice playback
  GET  /health         — stores + LLM reachability

Rate-limited with slowapi (README §12). Run:

    uvicorn api.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

logger = logging.getLogger("api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

UPLOAD_DIR = Path(__file__).resolve().parents[1] / "data" / "uploads"
SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_docs"

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app = FastAPI(title="MedClaim Agentic RAG", version="0.8.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_graph = None


def get_graph():
    global _graph
    if _graph is None:
        from agent.graph import build_graph

        _graph = build_graph()
    return _graph


@app.on_event("startup")
def warm() -> None:
    """Load every model once so first-request latency is generation-only."""
    try:
        from agent.nodes.grounding import _get_nli
        from agent.nodes.retrieve import get_retriever

        get_retriever()
        _get_nli()
        get_graph()
        logger.info("Warm start complete: retriever, NLI, graph loaded")
    except Exception:
        logger.exception("Warm start failed (endpoints may still work lazily)")


# ── Ask (SSE streaming) ─────────────────────────────────────────────────────
class AskBody(BaseModel):
    query: str
    source_type: Optional[str] = None
    thread_id: Optional[str] = None


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _final_payload(state: dict, thread_id: str) -> dict:
    by_id = {c["chunk_id"]: c for c in state.get("chunks", [])}
    citations = [
        {
            "chunk_id": cid,
            "doc_name": by_id.get(cid, {}).get("doc_name"),
            "doc_version": by_id.get(cid, {}).get("doc_version"),
            "section_title": by_id.get(cid, {}).get("section_title"),
            "page_number": by_id.get(cid, {}).get("page_number"),
            "has_bbox": bool(by_id.get(cid, {}).get("bbox")),
        }
        for cid in state.get("final_citations") or state.get("citations") or []
    ]
    return {
        "thread_id": thread_id,
        "status": state.get("status"),
        "answer": state.get("final_answer") or state.get("answer"),
        "citations": citations,
        "judge_score": state.get("judge_score"),
        "judge_reason": state.get("judge_reason"),
        "grounding_score": state.get("grounding_score"),
        "ungrounded_sentences": state.get("ungrounded_sentences", []),
        "pii_redacted": state.get("pii_redacted", []),
        "route": state.get("route"),
        # per-sentence best-grounding chunk (drives inline [n] citations)
        "sentence_attributions": state.get("sentence_attributions", []),
    }


@app.post("/ask")
@limiter.limit("10/minute")
def ask(request: Request, body: AskBody):
    """Stream the pipeline: token events live, then result/review."""
    thread_id = body.thread_id or uuid.uuid4().hex[:12]
    config = {"configurable": {"thread_id": thread_id}}
    graph = get_graph()

    def event_stream():
        yield _sse("start", {"thread_id": thread_id})
        try:
            for mode, chunk in graph.stream(
                {"query": body.query, "source_type": body.source_type},
                config, stream_mode=["custom", "updates"],
            ):
                if mode == "custom" and isinstance(chunk, dict) and "token" in chunk:
                    yield _sse("token", {"t": chunk["token"]})
                elif mode == "updates" and isinstance(chunk, dict):
                    for node in chunk:
                        if node != "__interrupt__":
                            yield _sse("status", {"node": node})
            state = graph.get_state(config).values
            if graph.get_state(config).next:  # paused at the HITL interrupt
                yield _sse("review", {
                    **_final_payload(state, thread_id),
                    "review_reason": state.get("review_reason"),
                })
            else:
                yield _sse("result", _final_payload(state, thread_id))
        except Exception as exc:  # surface, don't hang the stream
            logger.exception("ask stream failed")
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})


# ── HITL review ─────────────────────────────────────────────────────────────
class ReviewBody(BaseModel):
    verdict: str  # approved | edited | rejected
    answer: Optional[str] = None
    note: Optional[str] = None


@app.post("/review/{thread_id}")
@limiter.limit("30/minute")
def review(request: Request, thread_id: str, body: ReviewBody):
    from langgraph.types import Command

    if body.verdict not in ("approved", "edited", "rejected"):
        raise HTTPException(422, "verdict must be approved|edited|rejected")
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    if not graph.get_state(config).next:
        raise HTTPException(404, f"thread {thread_id} has no pending review")
    state = graph.invoke(
        Command(resume={"verdict": body.verdict, "answer": body.answer,
                        "note": body.note}),
        config,
    )
    return _final_payload(state, thread_id)


# ── Upload → ingestion ──────────────────────────────────────────────────────
@app.post("/upload")
@limiter.limit("5/minute")
def upload(request: Request, file: UploadFile = File(...),
           source_type: str = "policy"):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".pdf", ".docx", ".pptx"):
        raise HTTPException(422, f"unsupported file type {suffix}")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / Path(file.filename).name
    with open(dest, "wb") as fh:
        shutil.copyfileobj(file.file, fh)

    # Synchronous ingestion keeps the demo self-contained; swap to
    # ingest_document.delay(...) when a Celery worker is running.
    from ingestion.tasks import run_ingestion

    try:
        result = run_ingestion(str(dest), source_type)
    except Exception as exc:
        logger.exception("ingestion failed for %s", dest.name)
        raise HTTPException(500, f"ingestion failed: {exc}")
    return result


# ── Source highlighting (README §8 point 4) ────────────────────────────────
def _find_source_pdf(doc_name: str) -> Optional[Path]:
    for directory in (UPLOAD_DIR, SAMPLE_DIR):
        candidate = directory / doc_name
        if candidate.exists():
            return candidate
    return None


@app.get("/chunks/{chunk_id}")
@limiter.limit("60/minute")
def get_chunk(request: Request, chunk_id: str):
    from retrieval.vector_store import QdrantStore

    payloads = QdrantStore().retrieve([chunk_id])
    if chunk_id not in payloads:
        raise HTTPException(404, "chunk not found")
    return payloads[chunk_id]


@app.get("/source/{chunk_id}/image")
@limiter.limit("30/minute")
def source_image(request: Request, chunk_id: str, zoom: float = 2.0,
                 crop: int = 1):
    """Render the chunk's PDF page with its bbox regions highlighted.

    crop=1 (default) returns just the highlighted region plus padding — the
    UI lands the user ON the cited evidence instead of a full tall page they
    have to scroll. crop=0 returns the whole page.
    """
    import fitz  # PyMuPDF

    from retrieval.vector_store import QdrantStore

    payloads = QdrantStore().retrieve([chunk_id])
    if chunk_id not in payloads:
        raise HTTPException(404, "chunk not found")
    payload = payloads[chunk_id]
    doc_name, bboxes = payload.get("doc_name"), payload.get("bbox") or []
    if not doc_name or not bboxes:
        raise HTTPException(404, "chunk has no bbox provenance")
    pdf_path = _find_source_pdf(doc_name)
    if not pdf_path:
        raise HTTPException(404, f"source file {doc_name} not found on server")

    page_no = bboxes[0]["page_no"]
    pdf = fitz.open(pdf_path)
    try:
        page = pdf[page_no - 1]
        height = page.rect.height
        rects = []
        for bb in (b for b in bboxes if b["page_no"] == page_no):
            # Docling bboxes are BOTTOMLEFT-origin; PyMuPDF is TOPLEFT.
            if str(bb.get("coord_origin", "")).upper().endswith("BOTTOMLEFT"):
                rect = fitz.Rect(bb["l"], height - bb["t"], bb["r"], height - bb["b"])
            else:
                rect = fitz.Rect(bb["l"], bb["t"], bb["r"], bb["b"])
            rects.append(rect)
            page.draw_rect(rect, color=(0.95, 0.65, 0.1), fill=(1, 0.85, 0.3),
                           fill_opacity=0.25, width=1.5)
        clip = None
        if crop and rects:
            union = rects[0]
            for rect in rects[1:]:
                union |= rect
            pad = 36
            clip = fitz.Rect(max(0, union.x0 - pad), max(0, union.y0 - pad),
                             min(page.rect.x1, union.x1 + pad),
                             min(page.rect.y1, union.y1 + pad))
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
        return Response(content=pixmap.tobytes("png"), media_type="image/png")
    finally:
        pdf.close()


# ── TTS (free: edge-tts; browser SpeechSynthesis is the zero-latency path) ─
@app.get("/tts")
@limiter.limit("10/minute")
def tts(request: Request, text: str, voice: str = "en-US-AriaNeural"):
    import asyncio

    import edge_tts

    text = text[:1000]

    async def synth() -> bytes:
        buffer = io.BytesIO()
        communicate = edge_tts.Communicate(text, voice)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
        return buffer.getvalue()

    try:
        audio = asyncio.run(synth())
    except Exception as exc:
        raise HTTPException(502, f"TTS failed: {exc}")
    return Response(content=audio, media_type="audio/mpeg")


# ── Feedback (the self-healing flywheel, part 1: capture) ──────────────────
FEEDBACK_DB = Path(__file__).resolve().parents[1] / "data" / "feedback.db"


def _feedback_conn():
    import sqlite3

    FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(FEEDBACK_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT NOT NULL,
        rating INTEGER NOT NULL,          -- +1 / -1
        comment TEXT,
        query TEXT,
        answer TEXT,
        grounding_score REAL,
        judge_score REAL,
        created TEXT NOT NULL DEFAULT (datetime('now'))
    )""")
    return conn


class FeedbackBody(BaseModel):
    thread_id: str
    rating: int  # +1 or -1
    comment: Optional[str] = None


@app.post("/feedback")
@limiter.limit("30/minute")
def feedback(request: Request, body: FeedbackBody):
    """Record user feedback against the answered thread.

    Negative feedback (and reviewer-edited answers) are the raw material of
    the self-healing loop: scripts/export_feedback.py turns them into
    hard-case examples for the next fine-tune (v2 dataset).
    """
    if body.rating not in (1, -1):
        raise HTTPException(422, "rating must be +1 or -1")
    graph = get_graph()
    state = graph.get_state({"configurable": {"thread_id": body.thread_id}}).values
    if not state:
        raise HTTPException(404, f"unknown thread {body.thread_id}")
    with _feedback_conn() as conn:
        conn.execute(
            "INSERT INTO feedback (thread_id, rating, comment, query, answer,"
            " grounding_score, judge_score) VALUES (?,?,?,?,?,?,?)",
            (body.thread_id, body.rating, body.comment,
             state.get("query"), state.get("final_answer") or state.get("answer"),
             state.get("grounding_score"), state.get("judge_score")),
        )
    logger.info("feedback %+d on thread %s", body.rating, body.thread_id)
    return {"ok": True}


# ── Suggested follow-ups (lazy — fetched after the answer renders) ─────────
@app.get("/followups/{thread_id}")
@limiter.limit("20/minute")
def followups(request: Request, thread_id: str):
    from agent.nodes.generate import get_llm

    graph = get_graph()
    state = graph.get_state({"configurable": {"thread_id": thread_id}}).values
    if not state or not (state.get("final_answer") or state.get("answer")):
        raise HTTPException(404, f"no answer on thread {thread_id}")
    prompt = (
        "Given this claims-adjudication Q&A, suggest exactly 3 short follow-up "
        "questions the adjudicator would most likely ask next. Respond ONLY "
        'with JSON: {"questions": ["q1", "q2", "q3"]}.\n\n'
        f"Q: {state.get('query')}\nA: {state.get('final_answer') or state.get('answer')}"
    )
    try:
        raw = get_llm().chat(
            [{"role": "user", "content": prompt}],
            json_mode=True, temperature=0.4, max_tokens=150,
        )
        parsed = json.loads(raw)
        questions = [str(q).strip() for q in parsed.get("questions", []) if str(q).strip()]
    except Exception as exc:
        logger.warning("followups failed: %s", exc)
        questions = []
    return {"questions": questions[:3]}


# ── Health ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    from agent.nodes.generate import get_llm
    from retrieval.vector_store import QdrantStore

    checks = {}
    try:
        checks["qdrant_points"] = QdrantStore().count()
    except Exception as exc:
        checks["qdrant_points"] = f"error: {exc}"
    checks["llm_available"] = get_llm().is_available()
    return checks
