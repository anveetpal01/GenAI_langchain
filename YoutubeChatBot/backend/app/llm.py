"""Provider-agnostic chat model factory (SRS FR-GEN-01, FR-API-08, C-1)."""

from __future__ import annotations

import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.config import DEFAULT_MODELS, PROVIDER_KEY_ENV, get_settings


class LLMConfigError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def resolve(provider: str | None, model: str | None) -> tuple[str, str]:
    s = get_settings()
    provider = (provider or s.llm_provider).strip().lower()
    if provider not in DEFAULT_MODELS:
        raise LLMConfigError(
            "unknown_provider", f"Unknown LLM provider '{provider}'. Use one of: {', '.join(DEFAULT_MODELS)}."
        )
    if not model:
        model = s.llm_model if provider == s.llm_provider else DEFAULT_MODELS[provider]
    return provider, model


def has_key(provider: str, api_key: str | None = None) -> bool:
    env = PROVIDER_KEY_ENV[provider]
    return env is None or bool(api_key) or bool(os.environ.get(env))


@lru_cache(maxsize=16)
def _build(provider: str, model: str, api_key: str | None) -> BaseChatModel:
    s = get_settings()
    # Few retries: when a provider is overloaded, a fast clear error beats a minute of silence.
    kwargs = {"temperature": s.llm_temperature, "max_retries": s.llm_max_retries}
    if api_key:
        kwargs["api_key"] = api_key  # BYOK; otherwise the provider reads its env var
    return init_chat_model(model, model_provider=provider, **kwargs)


def get_chat_model(
    provider: str | None = None, model: str | None = None, api_key: str | None = None
) -> tuple[BaseChatModel, str, str]:
    provider, model = resolve(provider, model)
    if not has_key(provider, api_key):
        raise LLMConfigError(
            "missing_api_key",
            f"No API key for {provider}. Set {PROVIDER_KEY_ENV[provider]} in .env or add it in the extension options.",
        )
    return _build(provider, model, api_key), provider, model


def message_text(content) -> str:
    """Text of a (streamed) message; some providers return a list of parts."""
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "".join(parts)
