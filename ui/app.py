"""Streamlit UI for MedClaim (roadmap step 8, README §8 point 4).

Features:
  * chat with TRUE token streaming (SSE — the answer renders as it generates)
  * INLINE citations: [n] markers per sentence, attributed by the same NLI
    model that runs the grounding guardrail; numbered sources panel with the
    actual PDF page rendered and the cited region highlighted (bbox)
  * persistent chat HISTORY (SQLite): new / switch / delete, survives refresh
  * per-answer FEEDBACK (👍/👎 + comment) → /feedback — the self-healing
    flywheel (scripts/export_feedback.py turns 👎 into fine-tune hard cases)
  * suggested FOLLOW-UP question chips (lazy fetch, never blocks the stream)
  * quality badges (grounding / judge / route / PII), HITL review card,
    document upload, voice (browser SpeechSynthesis or server edge-tts)

Run (backend first):
    uvicorn api.main:app --port 8000
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ui import chat_store  # noqa: E402

API = os.getenv("MEDCLAIM_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="MedClaim", page_icon="🏥", layout="wide")

ss = st.session_state
ss.setdefault("chat_id", None)
ss.setdefault("messages", [])
ss.setdefault("pending_review", None)
ss.setdefault("queued_prompt", None)
ss.setdefault("followups", {})   # thread_id -> [questions]
ss.setdefault("feedback_sent", set())


# ── helpers ─────────────────────────────────────────────────────────────────
def sse_events(response: httpx.Response):
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
    ss._last_meta = None
    with httpx.stream("POST", f"{API}/ask",
                      json={"query": query, "source_type": source_type},
                      timeout=300.0) as response:
        response.raise_for_status()
        for event, data in sse_events(response):
            if event == "token":
                yield data["t"]
            elif event in ("result", "review"):
                data["_event"] = event
                ss._last_meta = data
            elif event == "error":
                yield f"\n\n⚠️ {data.get('detail')}"


def md_safe(text: str) -> str:
    """Escape $ so Streamlit's markdown doesn't render '$680.00 ... $136' as
    LaTeX math (observed live: policy amounts turned whole passages into
    code-style blocks)."""
    return (text or "").replace("$", "\\$")


def inline_cited_markdown(meta: dict) -> str:
    """Rebuild the answer with [n] inline citation markers per sentence,
    using the NLI sentence→chunk attributions computed by the grounding node."""
    answer = meta.get("answer") or ""
    attributions = meta.get("sentence_attributions") or []
    citations = meta.get("citations") or []
    numbering = {c["chunk_id"]: i + 1 for i, c in enumerate(citations)}
    if not attributions or not numbering:
        return answer
    parts = []
    for att in attributions:
        marker = ""
        n = numbering.get(att.get("chunk_id"))
        if n:
            marker = f" **[{n}]**"
        parts.append(att["sentence"].rstrip() + marker)
    return " ".join(parts)


def speak_browser(text: str) -> None:
    safe = json.dumps(text[:800])
    components.html(
        f"""<script>
        var u = new SpeechSynthesisUtterance({safe});
        u.rate = 1.05; window.speechSynthesis.cancel();
        window.speechSynthesis.speak(u);
        </script>""",
        height=0,
    )


def render_meta(meta: dict, message_key: str) -> None:
    # End users see: PII notice + sources. Scores/route/ungrounded details are
    # adjudicator diagnostics, hidden behind the sidebar toggle.
    if meta.get("pii_redacted"):
        st.warning(f"PII redacted from your question: {', '.join(meta['pii_redacted'])}")

    if ss.get("show_diagnostics"):
        badge_cols = st.columns(4)
        grounding = meta.get("grounding_score")
        judge = meta.get("judge_score")
        badge_cols[0].metric("Grounding", "—" if grounding is None else f"{grounding:.2f}")
        badge_cols[1].metric("Judge (advisory)", "—" if judge is None else f"{judge:.2f}")
        badge_cols[2].metric("Route", meta.get("route") or "—")
        badge_cols[3].metric("Status", meta.get("status") or "review")
        if meta.get("ungrounded_sentences"):
            st.error("Ungrounded sentences flagged: "
                     + " | ".join(meta["ungrounded_sentences"][:3]))

    citations = meta.get("citations") or []
    if citations:
        st.caption("SOURCES")
        for i, c in enumerate(citations, start=1):
            label = (f"[{i}] 📄 {c.get('doc_name')} v{c.get('doc_version')} — "
                     f"{c.get('section_title') or 'section'}"
                     + (f" (p.{c['page_number']})" if c.get("page_number") else ""))
            with st.expander(label):
                if c.get("has_bbox"):
                    # Cropped to the cited region — the user lands ON the
                    # evidence, no scrolling through a full page.
                    st.image(f"{API}/source/{c['chunk_id']}/image",
                             caption=f"cited region — {c.get('doc_name')} "
                                     f"p.{c.get('page_number')}")
                    if ss.get("show_diagnostics"):
                        st.image(f"{API}/source/{c['chunk_id']}/image?crop=0",
                                 caption="full page")
                else:
                    try:
                        chunk = httpx.get(f"{API}/chunks/{c['chunk_id']}",
                                          timeout=30).json()
                        st.text(chunk.get("text", "")[:1200])
                    except Exception:
                        st.caption("chunk text unavailable")

    # ── feedback row (self-healing flywheel) ────────────────────────────────
    thread_id = meta.get("thread_id")
    if thread_id and meta.get("status") == "answered":
        fb_key = f"fb_{message_key}"
        if fb_key in ss.feedback_sent:
            st.caption("✔ feedback recorded — thank you")
        else:
            c1, c2, c3 = st.columns([1, 1, 6])
            up = c1.button("👍", key=f"up_{message_key}")
            down = c2.button("👎", key=f"down_{message_key}")
            comment = c3.text_input("optional: what was wrong / right?",
                                    key=f"cmt_{message_key}",
                                    label_visibility="collapsed",
                                    placeholder="optional comment…")
            if up or down:
                try:
                    httpx.post(f"{API}/feedback",
                               json={"thread_id": thread_id,
                                     "rating": 1 if up else -1,
                                     "comment": comment or None}, timeout=30)
                    ss.feedback_sent.add(fb_key)
                    st.rerun()
                except Exception as exc:
                    st.error(f"feedback failed: {exc}")

    # ── follow-up chips (lazy) ──────────────────────────────────────────────
    if thread_id and meta.get("status") == "answered":
        if thread_id not in ss.followups:
            try:
                data = httpx.get(f"{API}/followups/{thread_id}", timeout=60).json()
                ss.followups[thread_id] = data.get("questions", [])
            except Exception:
                ss.followups[thread_id] = []
        questions = ss.followups.get(thread_id) or []
        if questions:
            st.caption("SUGGESTED FOLLOW-UPS")
            cols = st.columns(len(questions))
            for col, q in zip(cols, questions):
                if col.button(q, key=f"fu_{message_key}_{hash(q) & 0xffff}"):
                    ss.queued_prompt = q
                    st.rerun()


def run_turn(prompt: str, source_type) -> None:
    if ss.chat_id is None:
        ss.chat_id = chat_store.create_chat(prompt)
    chat_store.append_message(ss.chat_id, "user", prompt)
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(md_safe(prompt))
    with st.chat_message("assistant"):
        st.write_stream(ask_streaming(prompt, source_type))
        meta = ss.get("_last_meta") or {}
        if meta.get("_event") == "review":
            ss.pending_review = meta
            content = f"*(draft held for review)* {meta.get('answer', '')}"
            chat_store.append_message(ss.chat_id, "assistant", content)
            ss.messages.append({"role": "assistant", "content": content, "meta": None})
        else:
            content = inline_cited_markdown(meta) or meta.get("answer") or ""
            chat_store.append_message(ss.chat_id, "assistant", content, meta)
            ss.messages.append({"role": "assistant", "content": content, "meta": meta})
            if ss.get("voice_mode", "off").startswith("browser"):
                speak_browser(meta.get("answer") or content)
    st.rerun()


# ── sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏥 MedClaim")
    try:
        health = httpx.get(f"{API}/health", timeout=10).json()
        st.success(f"API up · {health.get('qdrant_points', '?')} chunks · "
                   f"LLM {'✓' if health.get('llm_available') else '✗'}")
    except Exception:
        st.error(f"API unreachable at {API}\n`uvicorn api.main:app --port 8000`")

    # ── history management ─────────────────────────────────────────────────
    st.subheader("💬 Chats")
    if st.button("➕ New chat", use_container_width=True):
        ss.chat_id, ss.messages, ss.pending_review = None, [], None
        st.rerun()
    for chat in chat_store.list_chats()[:15]:
        col_open, col_del = st.columns([5, 1])
        selected = chat["chat_id"] == ss.chat_id
        if col_open.button(("● " if selected else "") + chat["title"][:34],
                           key=f"open_{chat['chat_id']}", use_container_width=True):
            ss.chat_id = chat["chat_id"]
            ss.messages = chat_store.load_messages(chat["chat_id"])
            ss.pending_review = None
            st.rerun()
        if col_del.button("🗑", key=f"del_{chat['chat_id']}"):
            chat_store.delete_chat(chat["chat_id"])
            if selected:
                ss.chat_id, ss.messages = None, []
            st.rerun()

    st.divider()
    source_type = st.selectbox("Filter by source type",
                               [None, "policy", "clinical_guideline", "claim_note"],
                               format_func=lambda v: v or "all documents")
    ss.voice_mode = st.radio("Voice", ["off", "browser (instant)", "server (edge-tts)"])
    ss.show_diagnostics = st.toggle("🔧 Adjudicator diagnostics", value=False,
                                    help="Show grounding/judge scores, routing, "
                                         "and full source pages")

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

# ── main: document library ──────────────────────────────────────────────────
with st.expander("📚 Document Library — browse what's indexed"):
    lib_source_type = st.selectbox(
        "Filter by source type",
        [None, "policy", "clinical_guideline", "claim_note"],
        format_func=lambda v: v or "all documents",
        key="lib_source_type",
    )
    try:
        params = {"source_type": lib_source_type} if lib_source_type else {}
        docs = httpx.get(f"{API}/documents", params=params, timeout=30).json()
    except Exception as exc:
        docs = []
        st.error(f"couldn't load document list: {exc}")

    if not docs:
        st.caption("No indexed documents"
                   + (f" of type '{lib_source_type}'" if lib_source_type else "") + ".")

    type_icon = {"policy": "📋", "clinical_guideline": "🩺", "claim_note": "🗂️"}
    for doc in docs:
        icon = type_icon.get(doc.get("source_type"), "📄")
        title = (f"{icon} {doc.get('doc_name') or doc['doc_id']} · "
                 f"v{doc.get('doc_version')} · {doc.get('num_chunks')} chunks")
        with st.expander(title):
            st.caption(f"source_type: `{doc.get('source_type')}`"
                      + (f" · effective: {doc['effective_date']}"
                         if doc.get("effective_date") else "")
                      + (f" · {doc['num_tables']} table(s)"
                         if doc.get("num_tables") else ""))
            if doc.get("sections"):
                st.caption("Sections: " + ", ".join(doc["sections"]))

            preview_key = f"show_preview_{doc['doc_id']}"
            if st.button("👁️ Preview page 1", key=f"btn_{doc['doc_id']}"):
                ss[preview_key] = not ss.get(preview_key, False)
            if ss.get(preview_key):
                try:
                    st.image(f"{API}/documents/{doc['doc_id']}/preview",
                             caption=doc.get("doc_name"))
                except Exception:
                    st.caption("Preview unavailable — non-PDF source, or the "
                              "original file isn't retained on the server.")

# ── main: chat ──────────────────────────────────────────────────────────────
st.header("Ask the policy corpus")

for idx, message in enumerate(ss.messages):
    with st.chat_message(message["role"]):
        st.markdown(md_safe(message["content"]))
        if message.get("meta"):
            render_meta(message["meta"], message_key=str(idx))

if ss.pending_review:
    meta = ss.pending_review
    with st.container(border=True):
        head_l, head_r = st.columns([3, 2])
        head_l.subheader("⏸️ Needs your review")
        reasons = [r.strip() for r in (meta.get("review_reason") or "").split(";") if r.strip()]
        head_r.markdown(" ".join(f"`{r}`" for r in reasons) or "`verification incomplete`")

        draft = (meta.get("answer") or "").strip()
        if draft:
            st.info(md_safe(draft))
        else:
            st.warning("The model did not produce a usable answer — write one "
                       "under *Edit*, or Reject.")

        cites = meta.get("citations") or []
        if cites:
            st.caption("Evidence retrieved: " + " · ".join(
                f"{c.get('section_title') or c.get('doc_name')}"
                + (f" p.{c['page_number']}" if c.get("page_number") else "")
                for c in cites[:4]))

        verdict, payload = None, {}
        edited = draft
        with st.expander("✏️ Edit before sending"):
            edited = st.text_area("Corrected answer", value=draft,
                                  label_visibility="collapsed")
        note = st.text_input("Reviewer note (optional)",
                             placeholder="why you approved / edited / rejected…")
        c1, c2, c3 = st.columns(3)
        if c1.button("✅ Approve", type="primary", use_container_width=True):
            verdict = "approved"
        if c2.button("✏️ Send edited", use_container_width=True):
            verdict, payload = "edited", {"answer": edited}
        if c3.button("❌ Reject", use_container_width=True):
            verdict = "rejected"
        if verdict:
            final = httpx.post(f"{API}/review/{meta['thread_id']}",
                               json={"verdict": verdict, "note": note, **payload},
                               timeout=60.0).json()
            content = inline_cited_markdown(final) or final.get("answer") or ""
            chat_store.append_message(ss.chat_id, "assistant", content, final)
            ss.messages.append({"role": "assistant", "content": content, "meta": final})
            ss.pending_review = None
            st.rerun()

if ss.queued_prompt:
    prompt, ss.queued_prompt = ss.queued_prompt, None
    run_turn(prompt, source_type)

if prompt := st.chat_input("e.g. What is the copay for an MRI of the brain?"):
    run_turn(prompt, source_type)
