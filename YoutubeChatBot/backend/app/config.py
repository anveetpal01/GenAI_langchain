"""All tunables in one place (SRS NFR-07). Override any field via env / .env.

Env files are read in this order (first one wins for a given key):
    backend/.env  ->  <repo root>/.env
API keys stay in the process environment, where the LangChain provider
packages read them (GROQ_API_KEY, GOOGLE_API_KEY, ...); they are never logged.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from app.chunking import DEFAULT_MAX_TOKENS, DEFAULT_OVERLAP_TOKENS

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent.parent

load_dotenv(BACKEND_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")

# Default model per provider, used when only the provider is overridden.
DEFAULT_MODELS = {
    "groq": "llama-3.3-70b-versatile",
    "google_genai": "gemini-3.5-flash",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
    "ollama": "llama3.1",
}

# Env var each provider package reads its key from (Ollama needs none).
PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
}


class Settings(BaseSettings):
    data_dir: Path = BACKEND_DIR / "data"
    # Empty = SQLite in data_dir. Live: postgresql+psycopg://user:pass@host/db
    database_url: str = ""

    # Indexing
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_query_prefix: str = "Represent this sentence for searching relevant passages: "
    chunk_max_tokens: int = DEFAULT_MAX_TOKENS
    chunk_overlap_tokens: int = DEFAULT_OVERLAP_TOKENS

    # Retrieval
    dense_k: int = 20
    bm25_k: int = 20
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    mmr_k: int = 12
    rerank_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-base"
    final_k: int = 5
    # Below this best query-chunk cosine, answer "not in the video" (FR-GEN-04).
    # The reranker's absolute scores are tiny on long caption chunks, so it
    # only orders results; dense cosine separates on/off-topic far better.
    min_similarity: float = 0.45

    # Generation
    llm_provider: str = "groq"
    llm_model: str = DEFAULT_MODELS["groq"]
    # USD per 1M tokens, [input, output]. Paid-tier list prices, so free-tier
    # usage still shows what it would cost at scale. Models missing here are
    # logged with cost NULL ("unpriced"). Override with LLM_PRICES='{"model": [in, out]}'.
    # Source: ai.google.dev/gemini-api/docs/pricing, checked 2026-10-28
    # (gemini-3.8-flash doubles on 2027-01-01). Groq didn't publish per-model prices there.
    llm_prices: dict[str, list[float]] = {
        "gemini-3.5-flash-lite": [0.30, 2.50],
        "gemini-3.8-flash": [0.75, 3.75],
        "gemini-2.5-flash": [0.30, 2.50],
        "gemini-2.5-flash-lite": [0.10, 0.40],
    }
    llm_temperature: float = 0.2
    llm_max_retries: int = 2
    context_token_budget: int = 3000  # transcript excerpts
    history_token_budget: int = 1500
    history_turns: int = 6

    # Semantic answer cache (app/cache.py explains the two thresholds)
    cache_enabled: bool = True
    cache_min_similarity: float = 0.85
    cache_min_keyword_overlap: float = 0.5
    cache_ttl_days: int = 30
    cache_max_per_video: int = 300

    # Guardrails
    max_message_chars: int = 2000
    rate_limit_per_min: int = 20

    # API
    # Empty = any chrome-extension:// origin (dev). Set your extension ID for release.
    allowed_extension_ids: list[str] = []


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
