"""Local embedding and reranker models (SRS C-2). Loaded once, lazily."""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_embedder() -> SentenceTransformer:
    name = get_settings().embedding_model
    logger.info("Loading embedding model %s", name)
    return SentenceTransformer(name, device="cpu")


@lru_cache
def get_reranker() -> CrossEncoder:
    name = get_settings().reranker_model
    logger.info("Loading reranker %s", name)
    return CrossEncoder(name, device="cpu")


def embed_documents(texts: list[str]) -> np.ndarray:
    # Normalised vectors: dot product == cosine similarity.
    return get_embedder().encode(
        texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True
    )


def embed_query(text: str) -> np.ndarray:
    prefix = get_settings().embedding_query_prefix
    return get_embedder().encode(
        prefix + text, normalize_embeddings=True, convert_to_numpy=True
    )


def rerank_scores(query: str, texts: list[str]) -> np.ndarray:
    """Relevance of each text to the query, 0..1 (sigmoid of the cross-encoder logit)."""
    if not texts:
        return np.array([])
    return get_reranker().predict([(query, t) for t in texts], convert_to_numpy=True)


def warm_up() -> None:
    embed_query("warm up")
    if get_settings().rerank_enabled:
        rerank_scores("warm up", ["warm up"])
