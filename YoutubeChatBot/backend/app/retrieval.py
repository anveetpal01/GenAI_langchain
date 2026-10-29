"""Hybrid retrieval: dense + BM25 -> RRF -> MMR -> cross-encoder rerank.

SRS FR-RET-01..03, 05. Every stage records its latency and candidate count so
retrieval quality and speed can be compared across configs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from app.chunking import Chunk
from app.config import get_settings
from app.indexing import VideoIndex, get_collection, tokenize
from app.models import embed_query, rerank_scores

logger = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float | None  # reranker score (0..1), None if reranking is off
    similarity: float | None = None  # cosine(query, chunk)


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    metrics: dict = field(default_factory=dict)

    @property
    def top_similarity(self) -> float:
        return max((c.similarity or 0.0 for c in self.chunks), default=0.0)


def dense_search(index: VideoIndex, query_vec: np.ndarray, k: int) -> list[int]:
    n = min(k, len(index.chunks))
    result = get_collection(index.video_id).query(
        query_embeddings=[query_vec.tolist()], n_results=n, include=[]
    )
    return [int(i) for i in result["ids"][0]]


def bm25_search(index: VideoIndex, query: str, k: int) -> list[int]:
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = index.bm25.get_scores(tokens)
    order = np.argsort(-scores)[:k]
    # Chunks with no term overlap score 0; they are noise, not candidates.
    return [int(i) for i in order if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int) -> list[int]:
    """score(d) = sum over lists of 1 / (k + rank). Rank-based, so BM25 and
    cosine scores never need to be put on the same scale."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def mmr(
    query_vec: np.ndarray, candidates: list[int], embeddings: np.ndarray, lam: float, top_n: int
) -> list[int]:
    """Greedy Maximal Marginal Relevance:
    pick argmax  lam * sim(q, d) - (1 - lam) * max sim(d, already picked)."""
    if len(candidates) <= top_n:
        return candidates
    cand = np.array(candidates)
    vecs = embeddings[cand]
    relevance = vecs @ query_vec
    pairwise = vecs @ vecs.T

    picked: list[int] = []  # positions in `cand`
    remaining = list(range(len(cand)))
    while remaining and len(picked) < top_n:
        if picked:
            redundancy = pairwise[np.ix_(remaining, picked)].max(axis=1)
        else:
            redundancy = np.zeros(len(remaining))
        mmr_scores = lam * relevance[remaining] - (1 - lam) * redundancy
        best = remaining[int(np.argmax(mmr_scores))]
        picked.append(best)
        remaining.remove(best)
    return [int(cand[p]) for p in picked]


def retrieve(index: VideoIndex, query: str, query_vec: np.ndarray | None = None) -> RetrievalResult:
    s = get_settings()
    metrics: dict = {}
    timer = time.perf_counter

    t = timer()
    if query_vec is None:
        query_vec = embed_query(query)
    dense = dense_search(index, query_vec, s.dense_k)
    metrics["dense_ms"] = round((timer() - t) * 1000)

    t = timer()
    lexical = bm25_search(index, query, s.bm25_k)
    metrics["bm25_ms"] = round((timer() - t) * 1000)

    fused = reciprocal_rank_fusion([dense, lexical], s.rrf_k)
    diverse = mmr(query_vec, fused, index.embeddings, s.mmr_lambda, s.mmr_k)
    metrics["candidates"] = {"dense": len(dense), "bm25": len(lexical), "fused": len(fused), "mmr": len(diverse)}

    def similarity(i: int) -> float:
        return round(float(index.embeddings[i] @ query_vec), 4)

    if s.rerank_enabled:
        t = timer()
        scores = rerank_scores(query, [index.chunks[i].text for i in diverse])
        ranked = sorted(zip(diverse, scores), key=lambda p: p[1], reverse=True)[: s.final_k]
        metrics["rerank_ms"] = round((timer() - t) * 1000)
        result = [ScoredChunk(index.chunks[i], round(float(sc), 6), similarity(i)) for i, sc in ranked]
    else:
        result = [ScoredChunk(index.chunks[i], None, similarity(i)) for i in diverse[: s.final_k]]

    metrics["ranked"] = [
        {"chunk": c.chunk.chunk_index, "rerank": c.score, "cosine": c.similarity} for c in result
    ]
    metrics["total_ms"] = metrics["dense_ms"] + metrics["bm25_ms"] + metrics.get("rerank_ms", 0)
    logger.info("Retrieval for %s: %s", index.video_id, metrics)
    return RetrievalResult(result, metrics)


def spread_chunks(index: VideoIndex, n: int) -> list[ScoredChunk]:
    """`n` chunks evenly spaced over the video, for summary-style questions."""
    total = len(index.chunks)
    if total <= n:
        picks = range(total)
    else:
        picks = sorted({round(i * (total - 1) / max(n - 1, 1)) for i in range(n)})
    return [ScoredChunk(index.chunks[i], None) for i in picks]
