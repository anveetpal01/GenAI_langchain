"""Per-video index: Chroma vectors + BM25 corpus + metadata (SRS FR-IDX-05..07).

On disk (under settings.data_dir):
    chroma/                  Chroma persistent store, one collection per video
    videos/<video_id>.json   metadata + chunks (the BM25 corpus is rebuilt from
                             these in milliseconds, so it is not pickled)

Indexing is idempotent: an index is reused when its metadata matches the
current embedding model and chunk params *and* its Chroma collection exists.
A missing collection (e.g. data/chroma deleted) makes the index count as not
indexed, so the next /index call rebuilds it. `force=True` always rebuilds,
e.g. after the creator updated the captions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from app.chunking import Chunk, chunk_transcript
from app.config import get_settings
from app.models import embed_documents
from app.transcript import Caption, TranscriptError, VideoTranscript, get_transcript

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have he her his i in is it its "
    "me my of on or our she so that the their them they this to was we were "
    "what when which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """BM25 tokenizer: lowercase words, stopwords removed."""
    return [t for t in re.findall(r"\w+", text.lower()) if t not in _STOPWORDS]


@dataclass
class VideoIndex:
    video_id: str
    title: str | None
    channel: str | None
    language_code: str
    is_generated: bool
    transcript_hash: str
    embedding_model: str
    chunk_params: dict
    created_at: str
    chunks: list[Chunk]
    # Raw captions, for "what is being said right now?" (finer than a chunk).
    # Empty for indexes built before captions were stored.
    captions: list[Caption] = field(repr=False)
    embeddings: np.ndarray = field(repr=False)
    bm25: BM25Okapi = field(repr=False)

    @property
    def duration_s(self) -> float:
        return self.chunks[-1].end_s if self.chunks else 0.0


# ---------------------------------------------------------------- storage

@lru_cache
def _chroma() -> chromadb.ClientAPI:
    path = get_settings().data_dir / "chroma"
    return chromadb.PersistentClient(
        path=str(path), settings=chromadb.Settings(anonymized_telemetry=False)
    )


def collection_name(video_id: str) -> str:
    # Chroma names must start and end with [a-zA-Z0-9]; video IDs may end in - or _.
    return f"yt_{video_id}_chunks"


def get_collection(video_id: str):
    return _chroma().get_collection(collection_name(video_id))


def _meta_path(video_id: str):
    path = get_settings().data_dir / "videos"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{video_id}.json"


def _current_params() -> dict:
    s = get_settings()
    return {"max_tokens": s.chunk_max_tokens, "overlap_tokens": s.chunk_overlap_tokens}


def _read_meta(video_id: str) -> dict | None:
    path = _meta_path(video_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _is_current(meta: dict) -> bool:
    return (
        meta["embedding_model"] == get_settings().embedding_model
        and meta["chunk_params"] == _current_params()
    )


def _has_collection(video_id: str) -> bool:
    try:
        get_collection(video_id)
        return True
    except Exception:
        return False


def _is_ready(meta: dict | None) -> bool:
    return bool(meta) and _is_current(meta) and _has_collection(meta["video_id"])


def chroma_heartbeat() -> bool:
    try:
        _chroma().heartbeat()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- job state

_jobs: dict[str, dict] = {}  # video_id -> {"state": "indexing"|"failed", ...}
_jobs_lock = threading.Lock()
_loaded: dict[str, VideoIndex] = {}


def get_status(video_id: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(video_id)
    if job and job["state"] == "indexing":
        return {"state": "indexing", "chunks": None, "language": None, "error": None}

    meta = _read_meta(video_id)
    if _is_ready(meta):
        return {
            "state": "ready",
            "chunks": meta["chunk_count"],
            "language": meta["language_code"],
            "title": meta.get("title"),
            "channel": meta.get("channel"),
            "error": None,
        }
    if job and job["state"] == "failed":
        return {"state": "failed", "chunks": None, "language": None, "error": job["error"]}
    return {"state": "not_indexed", "chunks": None, "language": None, "error": None}


def try_start_job(video_id: str) -> bool:
    """Mark a video as indexing. False if a job for it is already running."""
    with _jobs_lock:
        if _jobs.get(video_id, {}).get("state") == "indexing":
            return False
        _jobs[video_id] = {"state": "indexing"}
        return True


# ---------------------------------------------------------------- indexing

def _transcript_hash(transcript: VideoTranscript) -> str:
    payload = json.dumps([asdict(c) for c in transcript.captions], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fetch_video_details(video_id: str) -> dict:
    """Title and channel name via YouTube oEmbed (no API key needed). Best effort."""
    url = "https://www.youtube.com/oembed?" + urllib.parse.urlencode(
        {"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"}
    )
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.load(resp)
        return {"title": data.get("title"), "channel": data.get("author_name")}
    except Exception:
        logger.warning("Could not fetch video details for %s", video_id)
        return {"title": None, "channel": None}


def run_index_job(video_id: str, force: bool = False) -> None:
    """Background-task entry point. Call `try_start_job` first."""
    try:
        index_video(video_id, force=force)
        with _jobs_lock:
            _jobs.pop(video_id, None)
    except TranscriptError as e:
        with _jobs_lock:
            _jobs[video_id] = {"state": "failed", "error": {"code": e.code, "message": e.message}}
    except Exception:
        logger.exception("Indexing failed for %s", video_id)
        with _jobs_lock:
            _jobs[video_id] = {
                "state": "failed",
                "error": {"code": "internal_error", "message": "Indexing failed unexpectedly."},
            }


def index_video(video_id: str, force: bool = False) -> VideoIndex:
    settings = get_settings()
    if not force and _is_ready(_read_meta(video_id)):
        logger.info("Index for %s is current, skipping", video_id)
        return load_index(video_id)

    t0 = time.perf_counter()
    transcript = get_transcript(video_id)
    digest = _transcript_hash(transcript)
    chunks = chunk_transcript(
        transcript,
        max_tokens=settings.chunk_max_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    vectors = embed_documents([c.text for c in chunks])

    client = _chroma()
    try:
        client.delete_collection(collection_name(video_id))
    except Exception:
        pass  # did not exist
    collection = client.create_collection(collection_name(video_id))
    collection.add(
        ids=[str(c.chunk_index) for c in chunks],
        embeddings=vectors.tolist(),
        documents=[c.text for c in chunks],
        metadatas=[
            {"video_id": video_id, "chunk_index": c.chunk_index, "start_s": c.start_s, "end_s": c.end_s}
            for c in chunks
        ],
    )

    details = _fetch_video_details(video_id)
    new_meta = {
        "video_id": video_id,
        "title": details["title"],
        "channel": details["channel"],
        "language_code": transcript.language_code,
        "is_generated": transcript.is_generated,
        "transcript_hash": digest,
        "chunk_count": len(chunks),
        "embedding_model": settings.embedding_model,
        "chunk_params": _current_params(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": [asdict(c) for c in chunks],
        "captions": [asdict(c) for c in transcript.captions],
    }
    # Write to a temp file then rename, so a crash never leaves half a file.
    path = _meta_path(video_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(new_meta, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

    _loaded.pop(video_id, None)
    logger.info(
        "Indexed %s: %d chunks in %.1fs", video_id, len(chunks), time.perf_counter() - t0
    )
    return load_index(video_id)


def load_index(video_id: str) -> VideoIndex | None:
    """Ready index for a video, or None if it is missing or stale."""
    if video_id in _loaded:
        return _loaded[video_id]

    meta = _read_meta(video_id)
    if not _is_ready(meta):
        return None

    chunks = [Chunk(**c) for c in meta["chunks"]]
    stored = get_collection(video_id).get(include=["embeddings"])
    by_id = dict(zip(stored["ids"], stored["embeddings"]))
    embeddings = np.array([by_id[str(c.chunk_index)] for c in chunks], dtype=np.float32)

    index = VideoIndex(
        video_id=video_id,
        title=meta.get("title"),
        channel=meta.get("channel"),
        language_code=meta["language_code"],
        is_generated=meta["is_generated"],
        transcript_hash=meta["transcript_hash"],
        embedding_model=meta["embedding_model"],
        chunk_params=meta["chunk_params"],
        created_at=meta["created_at"],
        chunks=chunks,
        captions=[Caption(**c) for c in meta.get("captions", [])],
        embeddings=embeddings,
        bm25=BM25Okapi([tokenize(c.text) for c in chunks]),
    )
    _loaded[video_id] = index
    return index
