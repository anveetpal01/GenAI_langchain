"""Semantic answer cache (Phase 2, SRS D-13).

A cached answer is reused when a new question about the same video means the
same thing. "Means the same" needs two checks, measured on real pairs
(2026-10-28, bge-small):

    "What does the speaker say about anger?" vs
      "What is said about anger in this video?"     0.856   same question
      "What does the speaker say about fear?"       0.811   different
    "Summarize this video" vs
      "Summarize the first half of the video"       0.900   different!

No single cosine threshold separates these, so a hit needs
  1. cosine similarity >= cache_min_similarity, and
  2. the same content words (Jaccard of keyword stems >= cache_min_keyword_overlap),
     which catches anger/fear and whole/first half.
An identical question (after normalising case and punctuation) always hits.

Lookups are always filtered by video and scope, so each search covers at
most a few hundred rows; cosine is computed in numpy, which works the same on
SQLite (local) and Postgres (live) without a vector extension.
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
from sqlalchemy import delete, func, insert, select, update

from app.config import get_settings
from app.db import answer_cache, get_engine
from app.indexing import VideoIndex

logger = logging.getLogger(__name__)

# Words that don't change what a question asks about this video.
_FILLER = frozenset(
    "a an the this that these those is are was were be been do does did can could would should will "
    "what whats which who how why when where please me i you he she they it its his her their my "
    "in on of about to for from at by with and or video videos clip speaker speakers person "
    "say says said saying tell talk talks talked talking speak speaks speaking mention mentions "
    # "thoughts" stays a content word: it is a topic in many videos.
    "mentioned give show view views opinion opinions think thinks "
    # common Hinglish glue words
    "ke ka ki ko me mein se kya hai hain tha the baare bare is iss ye yeh wo woh kehta kehte "
    "kehti bolta bolte bolti batao bataiye".split()
)


@dataclass
class CacheHit:
    entry_id: str
    answer: str
    citations: list[dict]
    similarity: float
    matched_query: str
    source_request_id: str | None


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def keywords(text: str) -> frozenset[str]:
    """Content-word stems. Crude 5-letter prefix stemming is enough to make
    summary/summarize or views/view equal while keeping anger/fear apart."""
    return frozenset(w[:5] for w in normalize(text).split() if len(w) > 1 and w not in _FILLER)


def keyword_overlap(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def scope_for(index: VideoIndex, prompt_version: str) -> str:
    raw = f"{index.transcript_hash}|{prompt_version}|{index.embedding_model}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def _query_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def lookup(video_id: str, scope: str, query: str, query_vec: np.ndarray) -> CacheHit | None:
    s = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(days=s.cache_ttl_days)
    with get_engine().connect() as conn:
        rows = conn.execute(
            select(answer_cache).where(
                answer_cache.c.video_id == video_id,
                answer_cache.c.scope == scope,
                answer_cache.c.created_at >= cutoff,
            )
        ).all()
    if not rows:
        return None

    q_hash, q_keys = _query_hash(query), keywords(query)
    best, best_sim = None, -1.0
    for row in rows:
        if row.query_hash == q_hash:
            best, best_sim = row, 1.0
            break
        sim = float(np.frombuffer(row.embedding, dtype=np.float32) @ query_vec)
        if (
            sim >= s.cache_min_similarity
            and sim > best_sim
            and keyword_overlap(q_keys, frozenset(row.keywords.split())) >= s.cache_min_keyword_overlap
        ):
            best, best_sim = row, sim
    if best is None:
        return None

    with get_engine().begin() as conn:
        conn.execute(
            update(answer_cache)
            .where(answer_cache.c.id == best.id)
            .values(hits=answer_cache.c.hits + 1, last_hit_at=datetime.now(timezone.utc))
        )
    return CacheHit(best.id, best.answer, best.citations, round(best_sim, 4), best.query, best.source_request_id)


def save(
    video_id: str,
    scope: str,
    query: str,
    query_vec: np.ndarray,
    answer: str,
    citations: list[dict],
    source_request_id: str,
) -> None:
    """Store an answer. Never raises: a cache write must not break a reply."""
    s = get_settings()
    try:
        with get_engine().begin() as conn:
            conn.execute(
                insert(answer_cache).values(
                    id=uuid.uuid4().hex,
                    video_id=video_id,
                    scope=scope,
                    query=query,
                    query_hash=_query_hash(query),
                    keywords=" ".join(sorted(keywords(query))),
                    embedding=np.asarray(query_vec, dtype=np.float32).tobytes(),
                    answer=answer,
                    citations=citations,
                    source_request_id=source_request_id,
                    created_at=datetime.now(timezone.utc),
                    hits=0,
                )
            )
            # Keep each video's cache bounded: drop the least useful entries.
            count = conn.execute(
                select(func.count()).select_from(answer_cache).where(answer_cache.c.video_id == video_id)
            ).scalar_one()
            excess = count - s.cache_max_per_video
            if excess > 0:
                victims = select(answer_cache.c.id).where(answer_cache.c.video_id == video_id).order_by(
                    answer_cache.c.hits.asc(), answer_cache.c.created_at.asc()
                ).limit(excess)
                conn.execute(delete(answer_cache).where(answer_cache.c.id.in_(victims.scalar_subquery())))
    except Exception:
        logger.exception("Could not save cache entry for %s", video_id)
