"""Relational storage behind one engine (SRS D-13).

`DATABASE_URL` picks the backend: unset -> SQLite file in data_dir (local),
`postgresql+psycopg://...` -> Postgres (live). Tables are defined once here
with SQLAlchemy Core, so the same queries run on both.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Engine,
    Float,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
)

from app.config import get_settings

metadata = MetaData()

# Chat history (FR-MEM-01). Same shape as the original sqlite3 table, so
# existing local chat.db files keep working.
messages = Table(
    "messages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("video_id", String(32), nullable=False),
    Column("session_id", String(64), nullable=False),
    Column("role", String(16), nullable=False),
    Column("content", Text, nullable=False),
    Column("citations_json", Text),
    Column("prompt_version", String(32)),
    Column("latency_ms", Integer),
    Column("created_at", String(40), nullable=False),
    Index("idx_messages_session", "video_id", "session_id", "id"),
)

# One row per /chat request: what it cost and how long it took (NFR-05).
# `calls` holds the per-LLM-call breakdown (condense, qa).
requests = Table(
    "requests",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("video_id", String(32), nullable=False),
    Column("session_id", String(64), nullable=False),
    Column("status", String(16), nullable=False),  # ok | error | stopped
    Column("error_code", String(64)),
    Column("route", String(32), nullable=False),  # hybrid | summary | time_local | not_found
    Column("provider", String(32), nullable=False),
    Column("model", String(64), nullable=False),
    Column("prompt_version", String(32), nullable=False),
    Column("cache_hit", Boolean, nullable=False, default=False),
    Column("tokens_in", Integer, nullable=False, default=0),
    Column("tokens_out", Integer, nullable=False, default=0),
    Column("cost_usd", Float),  # NULL = a model without a known price
    Column("usage_estimated", Boolean, nullable=False, default=False),
    Column("latency_ms", Integer, nullable=False),
    Column("ttft_ms", Integer),  # time to first token
    Column("retrieval_ms", Integer),
    Column("llm_ms", Integer),
    Column("calls", JSON),
    Index("idx_requests_created", "created_at"),
    Index("idx_requests_video", "video_id", "created_at"),
)


# Semantic answer cache (Phase 2, SRS D-13). Shared by all users of a video
# (videos are public). `scope` ties an entry to the exact transcript, prompt
# version and embedding model it was produced with; any change = new scope.
answer_cache = Table(
    "answer_cache",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("video_id", String(32), nullable=False),
    Column("scope", String(64), nullable=False),
    Column("query", Text, nullable=False),
    Column("query_hash", String(64), nullable=False),  # normalised text, for exact hits
    Column("keywords", Text, nullable=False),  # space-separated content-word stems
    Column("embedding", LargeBinary, nullable=False),  # float32 bytes
    Column("answer", Text, nullable=False),
    Column("citations", JSON, nullable=False),
    Column("source_request_id", String(32)),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("last_hit_at", DateTime(timezone=True)),
    Column("hits", Integer, nullable=False, default=0),
    Index("idx_cache_lookup", "video_id", "scope"),
)


def database_url() -> str:
    s = get_settings()
    return s.database_url or f"sqlite:///{(s.data_dir / 'chat.db').as_posix()}"


@lru_cache
def get_engine() -> Engine:
    url = database_url()
    if url.startswith("sqlite"):
        # FastAPI serves requests from several threads.
        engine = create_engine(url, connect_args={"check_same_thread": False})
    else:
        engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)
    metadata.create_all(engine)
    return engine

