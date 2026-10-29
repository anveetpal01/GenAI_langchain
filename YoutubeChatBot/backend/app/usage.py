"""Token usage and cost of LLM calls, and the per-request log (Phase 0 of
the semantic cache, SRS D-13). Without these numbers the cache's saving
can't be shown.

Tokens come from the provider (LangChain `usage_metadata`). When a provider
doesn't report them, they are estimated with tiktoken and the row is marked
`usage_estimated`, so dashboards never mix guesses with facts silently.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from sqlalchemy import insert

from app.chunking import count_tokens
from app.config import get_settings
from app.db import get_engine, requests

logger = logging.getLogger(__name__)


@dataclass
class CallUsage:
    purpose: str  # condense | qa
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    estimated: bool
    cost_usd: float | None  # None = no price known for this model
    ms: int


def price_per_million(model: str) -> tuple[float, float] | None:
    """(input, output) USD per 1M tokens, from settings.llm_prices."""
    price = get_settings().llm_prices.get(model)
    return (price[0], price[1]) if price else None


def call_usage(
    purpose: str,
    provider: str,
    model: str,
    usage_metadata: dict | None,
    prompt_text: str,
    output_text: str,
    ms: int,
) -> CallUsage:
    if usage_metadata and usage_metadata.get("input_tokens") is not None:
        tokens_in = int(usage_metadata.get("input_tokens") or 0)
        tokens_out = int(usage_metadata.get("output_tokens") or 0)
        estimated = False
    else:
        tokens_in, tokens_out, estimated = count_tokens(prompt_text), count_tokens(output_text), True

    price = price_per_million(model)
    cost = None
    if price:
        cost = round((tokens_in * price[0] + tokens_out * price[1]) / 1_000_000, 8)
    return CallUsage(purpose, provider, model, tokens_in, tokens_out, estimated, cost, ms)


def total_cost(calls: list[CallUsage]) -> float | None:
    """Sum of call costs; None if any call has no known price (a partial
    sum would look like a real, too-low number)."""
    if any(c.cost_usd is None for c in calls):
        return None
    return round(sum(c.cost_usd or 0.0 for c in calls), 8)


def usage_summary(calls: list[CallUsage]) -> dict:
    return {
        "tokens_in": sum(c.tokens_in for c in calls),
        "tokens_out": sum(c.tokens_out for c in calls),
        "cost_usd": total_cost(calls),
        "estimated": any(c.estimated for c in calls),
    }


def log_request(
    *,
    request_id: str,
    video_id: str,
    session_id: str,
    status: str,
    error_code: str | None,
    route: str,
    provider: str,
    model: str,
    prompt_version: str,
    cache_hit: bool,
    calls: list[CallUsage],
    latency_ms: int,
    ttft_ms: int | None,
    retrieval_ms: int | None,
    llm_ms: int | None,
) -> None:
    """Write one row to `requests`. Never raises: losing a metrics row must
    not break a user's answer."""
    summary = usage_summary(calls)
    try:
        with get_engine().begin() as conn:
            conn.execute(
                insert(requests).values(
                    id=request_id,
                    created_at=datetime.now(timezone.utc),
                    video_id=video_id,
                    session_id=session_id,
                    status=status,
                    error_code=error_code,
                    route=route,
                    provider=provider,
                    model=model,
                    prompt_version=prompt_version,
                    cache_hit=cache_hit,
                    tokens_in=summary["tokens_in"],
                    tokens_out=summary["tokens_out"],
                    cost_usd=summary["cost_usd"],
                    usage_estimated=summary["estimated"],
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                    retrieval_ms=retrieval_ms,
                    llm_ms=llm_ms,
                    calls=[asdict(c) for c in calls],
                )
            )
    except Exception:
        logger.exception("Could not log request %s", request_id)
