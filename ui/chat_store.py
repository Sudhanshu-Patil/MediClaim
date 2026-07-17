"""Persistent chat history for the Streamlit UI (SQLite, survives refresh)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path

DB = Path(__file__).resolve().parents[1] / "data" / "chats.db"


def _conn() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chats (
            chat_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT NOT NULL REFERENCES chats(chat_id),
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            meta TEXT,
            created TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    return conn


def list_chats() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT chat_id, title, created FROM chats ORDER BY created DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def create_chat(title: str) -> str:
    chat_id = uuid.uuid4().hex[:10]
    with _conn() as conn:
        conn.execute("INSERT INTO chats (chat_id, title) VALUES (?, ?)",
                     (chat_id, title[:60] or "New chat"))
    return chat_id


def delete_chat(chat_id: str) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
        conn.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))


def load_messages(chat_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, meta FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
    return [{"role": r["role"], "content": r["content"],
             "meta": json.loads(r["meta"]) if r["meta"] else None} for r in rows]


def append_message(chat_id: str, role: str, content: str, meta: dict | None = None) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, meta) VALUES (?,?,?,?)",
            (chat_id, role, content, json.dumps(meta, ensure_ascii=False) if meta else None),
        )
