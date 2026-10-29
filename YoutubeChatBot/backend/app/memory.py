"""Per (video_id, session_id) chat history (SRS FR-MEM-01, data model §6).

Stored through app.db, so it is SQLite locally and Postgres live. A plain
table instead of LangChain's SQLChatMessageHistory, because we also keep
citations, prompt version and latency per message.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import delete, insert, select

from app.db import get_engine, messages


def add_message(
    video_id: str,
    session_id: str,
    role: str,
    content: str,
    citations: list[dict] | None = None,
    prompt_version: str | None = None,
    latency_ms: int | None = None,
) -> None:
    with get_engine().begin() as conn:
        conn.execute(
            insert(messages).values(
                video_id=video_id,
                session_id=session_id,
                role=role,
                content=content,
                citations_json=json.dumps(citations) if citations is not None else None,
                prompt_version=prompt_version,
                latency_ms=latency_ms,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        )


def get_history(video_id: str, session_id: str, limit: int | None = None) -> list[dict]:
    """Messages oldest-first; with `limit`, only the most recent ones."""
    query = (
        select(messages.c.role, messages.c.content, messages.c.citations_json, messages.c.created_at)
        .where(messages.c.video_id == video_id, messages.c.session_id == session_id)
        .order_by(messages.c.id.desc())
    )
    if limit is not None:
        query = query.limit(limit)
    with get_engine().connect() as conn:
        rows = conn.execute(query).all()
    return [
        {
            "role": r.role,
            "content": r.content,
            "citations": json.loads(r.citations_json) if r.citations_json else [],
            "created_at": r.created_at,
        }
        for r in reversed(rows)
    ]


def clear_history(video_id: str, session_id: str) -> int:
    with get_engine().begin() as conn:
        result = conn.execute(
            delete(messages).where(messages.c.video_id == video_id, messages.c.session_id == session_id)
        )
        return result.rowcount
