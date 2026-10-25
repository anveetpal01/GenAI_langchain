"""Per (video_id, session_id) chat history in SQLite (SRS FR-MEM-01, data model §6).

A plain sqlite3 store instead of LangChain's SQLChatMessageHistory, because
we also keep citations, prompt version and latency per message.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache

from app.config import get_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id       TEXT NOT NULL,
    session_id     TEXT NOT NULL,
    role           TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content        TEXT NOT NULL,
    citations_json TEXT,
    prompt_version TEXT,
    latency_ms     INTEGER,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (video_id, session_id, id);
"""


@lru_cache
def _db_path() -> str:
    path = get_settings().data_dir / "chat.db"
    with closing(sqlite3.connect(path)) as conn:
        conn.executescript(_SCHEMA)
    return str(path)


def _connect() -> sqlite3.Connection:
    # One short-lived connection per call: safe across FastAPI's worker threads.
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def add_message(
    video_id: str,
    session_id: str,
    role: str,
    content: str,
    citations: list[dict] | None = None,
    prompt_version: str | None = None,
    latency_ms: int | None = None,
) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO messages (video_id, session_id, role, content, citations_json,"
            " prompt_version, latency_ms, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                video_id, session_id, role, content,
                json.dumps(citations) if citations is not None else None,
                prompt_version, latency_ms, datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_history(video_id: str, session_id: str, limit: int | None = None) -> list[dict]:
    """Messages oldest-first; with `limit`, only the most recent ones."""
    sql = (
        "SELECT role, content, citations_json, created_at FROM messages"
        " WHERE video_id = ? AND session_id = ? ORDER BY id DESC"
    )
    params: tuple = (video_id, session_id)
    if limit is not None:
        sql += " LIMIT ?"
        params += (limit,)
    with closing(_connect()) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {
            "role": r["role"],
            "content": r["content"],
            "citations": json.loads(r["citations_json"]) if r["citations_json"] else [],
            "created_at": r["created_at"],
        }
        for r in reversed(rows)
    ]


def clear_history(video_id: str, session_id: str) -> int:
    with closing(_connect()) as conn, conn:
        cur = conn.execute(
            "DELETE FROM messages WHERE video_id = ? AND session_id = ?", (video_id, session_id)
        )
        return cur.rowcount
