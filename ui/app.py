"""Streamlit UI for MedClaim (roadmap step 8, README §8 point 4).

Features:
  * chat with TRUE token streaming (SSE from the FastAPI backend — the answer
    renders as the model generates it)
  * per-citation source panel: doc/version/section/page plus the actual PDF
    page rendered with the cited region highlighted (bbox provenance)
  * quality badges per answer: grounding score, judge score, route, PII
  * document upload → live ingestion into the corpus
  * HITL review queue: approve / edit / reject paused answers
  * voice: browser SpeechSynthesis (instant, free) or server edge-tts audio

Run (backend must be up first):
    uvicorn api.main:app --port 8000
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st
import streamlit.components.v1 as components

API = os.getenv("MEDCLAIM_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="MedClaim", page_icon="🏥", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content, meta}]
if "pending_review" not in st.session_state:
    st.session_state.pending_review = None


# ── helpers ─────────────────────────────────────────────────────────────────
def sse_events(response: httpx.Response):
    """Parse an SSE byte stream into (event, data) tuples."""
    event, data_lines = None, []
    for line in response.iter_lines():
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: "):
            data_lines.append(line[6:])
        elif not line and event:
            yield event, json.loads("\n".join(data_lines) or "{}")
            event, data_lines = None, []


def ask_streaming(query: str, source_type):
    """Yield answer tokens for st.write_stream; stash the final meta."""
    st.session_state._last_meta = None
    with httpx.stream("POST", f"{API}/ask",
                      json={"query": query, "source_type": source_type},
                      timeout=300.0) as response:
        response.raise_for_status()
        for event, data in sse_events(response):
            if event == "token":
                yield data["t"]
            elif event in ("result", "review"):
                data["_event"] = event
                st.session_state._last_meta = data
            elif event == "error":
                yield f"\n\n⚠️ {data.get('detail')}"


def speak_browser(text: str, key: str) -> None:
    """Zero-cost TTS via the browser's SpeechSynthesis API."""
    safe = json.dumps(text[:800])
    components.html(
        f"""<script>
        var u = new SpeechSynthesisUtterance({safe});
        u.rate = 1.05; window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
        </script>""",
        height=0,
    )


def render_meta(meta: dict) -> None:
    """Quality badges + citation source panel."""
    badge_cols = st.columns(4)
    grounding = meta.get("grounding_score")
    judge = meta.get("judge_score")
    badge_cols[0].metric("Grounding", "—" if grounding is None else f"{grounding:.2f}")
    badge_cols[1].metric("Judge", "—" if judge is None else f"{judge:.2f}")
    badge_cols[2].metric("Route", meta.get("route") or "—")
    badge_cols[3].metric("Status", meta.get("status") or "review")
    if meta.get("pii_redacted"):
        st.warning(f"PII redacted from your question: {', '.join(meta['pii_redacted'])}")
    if meta.get("ungrounded_sentences"):
        st.error("Ungrounded sentences flagged: "
                 + " | ".join(meta["ungrounded_sentences"][:3]))

    citations = meta.get("citations") or []
    if citations:
        st.caption("SOURCES (grouped by document)")
        by_doc: dict[str, list[dict]] = {}
        for c in citations:
            by_doc.setdefault(c.get("doc_name") or "unknown", []).append(c)
        for doc_name, cites in by_doc.items():
            for c in cites:
                label = (f"📄 {doc_name} v{c.get('doc_version')} — "
                         f"{c.get('section_title') or 'section'}"
                         + (f" (p.{c['page_number']})" if c.get("page_number") else ""))
                with st.expander(label):
                    try:
                        chunk = httpx.get(f"{API}/chunks/{c['chunk_id']}", timeout=30).json()
                        st.text(chunk.get("text", "")[:1200])
                    except Exception:
                        st.caption("chunk text unavailable")
                    if c.get("has_bbox"):
                        st.image(f"{API}/source/{c['chunk_id']}/image",
                                 caption="source page, cited region highlighted")


# ── sidebar: corpus + settings ──────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 MedClaim")
    st.caption("Local agentic RAG for claims adjudication")

    try:
        health = httpx.get(f"{API}/health", timeout=10).json()
        st.success(f"API up · {health.get('qdrant_points', '?')} chunks · "
                   f"LLM {'✓' if health.get('llm_available') else '✗'}")
    except Exception:
        st.error(f"API unreachable at {API} — start it:\n`uvicorn api.main:app --port 8000`")

    source_type = st.selectbox("Filter by source type",
                               [None, "policy", "clinical_guideline", "claim_note"],
                               format_func=lambda v: v or "all documents")
    voice_mode = st.radio("Voice", ["off", "browser (instant)", "server (edge-tts)"])

    st.divider()
    st.subheader("📥 Add a document")
    uploaded = st.file_uploader("PDF / DOCX / PPTX", type=["pdf", "docx", "pptx"])
    upload_type = st.selectbox("source_type", ["policy", "clinical_guideline", "claim_note"])
    if uploaded and st.button("Ingest", use_container_width=True):
        with st.spinner("Parsing, chunking, embedding…"):
            result = httpx.post(
                f"{API}/upload", params={"source_type": upload_type},
                files={"file": (uploaded.name, uploaded.getvalue())},
                timeout=600.0,
            )
        if result.status_code == 200:
            info = result.json()
            st.success(f"{info.get('action')}: {info.get('chunks', 0)} chunks, "
                       f"v{info.get('doc_version')}")
        else:
            st.error(result.text[:300])

# ── main: chat ──────────────────────────────────────────────────────────────
st.header("Ask the policy corpus")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            render_meta(message["meta"])

if st.session_state.pending_review:
    meta = st.session_state.pending_review
    with st.container(border=True):
        st.subheader("⏸️ Human review required")
        st.caption(meta.get("review_reason") or "")
        st.markdown(f"**Draft:** {meta.get('answer')}")
        edited = st.text_area("Edit the answer (for verdict=edited)",
                              value=meta.get("answer") or "")
        note = st.text_input("Reviewer note")
        c1, c2, c3 = st.columns(3)
        verdict = None
        if c1.button("✅ Approve"):
            verdict, payload = "approved", {}
        if c2.button("✏️ Send edited"):
            verdict, payload = "edited", {"answer": edited}
        if c3.button("❌ Reject"):
            verdict, payload = "rejected", {}
        if verdict:
            final = httpx.post(f"{API}/review/{meta['thread_id']}",
                               json={"verdict": verdict, "note": note, **payload},
                               timeout=60.0).json()
            st.session_state.messages.append(
                {"role": "assistant", "content": final.get("answer") or "",
                 "meta": final})
            st.session_state.pending_review = None
            st.rerun()

if prompt := st.chat_input("e.g. What is the copay for an MRI of the brain?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        streamed_text = st.write_stream(ask_streaming(prompt, source_type))
        meta = st.session_state.get("_last_meta") or {}
        if meta.get("_event") == "review":
            st.session_state.pending_review = meta
            st.info("Held for human review — see panel above after rerun.")
            st.session_state.messages.append(
                {"role": "assistant",
                 "content": f"*(draft held for review)* {meta.get('answer','')}",
                 "meta": None})
        else:
            final_text = meta.get("answer") or streamed_text
            render_meta(meta)
            st.session_state.messages.append(
                {"role": "assistant", "content": final_text, "meta": meta})
            if voice_mode.startswith("browser"):
                speak_browser(final_text, key=str(len(st.session_state.messages)))
            elif voice_mode.startswith("server"):
                st.audio(f"{API}/tts?text={httpx.QueryParams({'text': final_text[:800]})['text']}",
                         format="audio/mpeg")
    st.rerun()
