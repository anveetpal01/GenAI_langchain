"""Chat orchestration: guardrails -> condense -> retrieve -> prompt -> stream -> cite.

`prepare_chat` does every check that can fail fast (so the API can return a
proper HTTP error); `stream_chat` then yields SSE events:
    token      {"text": "..."}
    citations  {"citations": [...]}
    done       {"answer": <cleaned answer>, "low_confidence": bool, ...}
    error      {"code": "...", "message": "..."}
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app import cache, guardrails, memory
from app.chunking import Chunk, count_tokens, format_ts
from app.config import get_settings
from app.indexing import VideoIndex, load_index
from app.llm import get_chat_model, message_text
from app.models import embed_query
from app.prompts import CONDENSE_PROMPT, NOT_FOUND_ANSWER, PROMPT_VERSION, QA_TEMPLATE, SYSTEM_PROMPT
from app.retrieval import ScoredChunk, retrieve, spread_chunks
from app.usage import CallUsage, call_usage, log_request, usage_summary

logger = logging.getLogger(__name__)

# Summary-style questions match weakly against any single chunk, so they get
# excerpts spread over the whole video instead (a v1 stand-in for FR-RET-07).
_SUMMARY_REQUEST = re.compile(
    r"\b(summar\w*|overview|tl;?dr|key (takeaways|points)|main (points|ideas)|"
    r"what('?s| is) (this|the) video about)\b",
    re.IGNORECASE,
)

# "What is being said right now?" has no topical words to retrieve with, so the
# transcript around the playhead is put first (FR-AUG-04, FR-UI-08).
_TIME_LOCAL_REQUEST = re.compile(
    r"\b(right now|just now|at this (point|moment|time)|currently|just said|being said now|abhi)\b",
    re.IGNORECASE,
)
# Questions about the video itself (title, channel, who is in it) match no
# transcript chunk, so the "not in the video" shortcut must not answer them:
# the video details in the prompt can. Patterns, not bare words: "What does
# the speaker say about anger?" is about the content, "Who is the speaker?"
# is not. Videos differ (lecture, podcast, song, news, gaming), so the
# people words cover more than "speaker". A question this misses only gets
# "not found"; it never gets a wrong answer.
_PEOPLE = (
    r"speakers?|hosts?|guests?|singers?|artists?|band|rappers?|narrators?|presenters?|teachers?|"
    r"instructors?|lecturers?|professors?|anchors?|youtubers?|creators?|authors?|interviewers?|"
    r"commentators?|players?|actors?|actress|comedians?|person|people|man|woman|guy|girl|boy"
)
_VIDEO_DETAILS_REQUEST = re.compile(
    # who is the singer / who's the guest / who are the speakers
    rf"\bwho(?:'s|\s+is|\s+are|\s+was|\s+were)\s+(?:the\s+|this\s+|that\s+)?(?:main\s+)?(?:{_PEOPLE})\b"
    # who is speaking / who is singing / who is in this video
    r"|\bwho(?:'s|\s+is|\s+are)?\s+(?:speaking|talking|singing|playing|narrating|presenting|hosting|"
    r"rapping|in\s+(?:this|the)\s+video)\b"
    # who made / uploaded / sang this
    r"|\bwho\s+(?:made|makes|uploaded|posted|published|created|produced|directed|sang|sings|wrote)\b"
    # the video's own fields
    r"|\b(?:title|channel|uploaded\s+by|upload\s+date|duration)\b"
    r"|\bhow\s+long\s+(?:is|was)\s+(?:this|the|it)\b|\blength\s+of\s+(?:this|the)\s+video\b"
    r"|\bname\s+of\s+(?:the|this)\s+(?:song|video|singer|speaker|channel)\b"
    # Hinglish: gaane ka naam kya hai? / ye kisne banaya?
    r"|\b(?:song|gaane|gaana|video|channel)\s+(?:ka|ki)\s+naam\b|\bkisne\s+(?:banaya|upload|gaaya)",
    re.IGNORECASE,
)
_LOCAL_BEFORE_S = 45  # what was said leading up to the playhead
_LOCAL_AFTER_S = 20


def _excerpt_at(index: VideoIndex, seconds: float) -> ScoredChunk | None:
    """The captions around `seconds` as one excerpt. A whole chunk spans
    ~2-3 minutes, too coarse for "right now"; falls back to it for indexes
    built without captions."""
    covering = [c for c in index.chunks if c.start_s <= seconds <= c.end_s]
    if not covering:
        return None
    nearest = min(covering, key=lambda c: abs((c.start_s + c.end_s) / 2 - seconds))

    window = [
        c for c in index.captions
        if c.end_s >= seconds - _LOCAL_BEFORE_S and c.start_s <= seconds + _LOCAL_AFTER_S
    ]
    if not window:
        return ScoredChunk(nearest, None)
    text = " ".join(c.text for c in window)
    excerpt = Chunk(
        video_id=index.video_id,
        chunk_index=nearest.chunk_index,
        text=text,
        start_s=window[0].start_s,
        end_s=max(c.end_s for c in window),
        token_count=count_tokens(text),
    )
    return ScoredChunk(excerpt, None)


class ChatError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


@dataclass
class ChatEvent:
    event: str
    data: dict


@dataclass
class ChatContext:
    video_id: str
    session_id: str
    message: str
    playback_time: float | None
    index: VideoIndex
    llm: BaseChatModel
    provider: str
    model: str
    use_cache: bool = True


def prepare_chat(
    video_id: str,
    session_id: str,
    message: str,
    playback_time: float | None = None,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    use_cache: bool = True,
) -> ChatContext:
    """Validate everything up front. Raises GuardrailError / ChatError / LLMConfigError."""
    message = guardrails.clean_message(message)
    guardrails.rate_limiter.check(session_id)

    index = load_index(video_id)
    if index is None:
        raise ChatError("not_indexed", "This video is not indexed yet. Index it first.", status=409)

    llm, provider, model = get_chat_model(provider, model, api_key)
    return ChatContext(
        video_id, session_id, message, playback_time, index, llm, provider, model,
        use_cache=use_cache and get_settings().cache_enabled,
    )


# ---------------------------------------------------------------- prompt building

def _fit_history(history: list[dict], budget: int) -> list[dict]:
    """Keep the most recent turns that fit the token budget (oldest dropped first)."""
    kept, used = [], 0
    for msg in reversed(history):
        cost = count_tokens(msg["content"])
        if used + cost > budget:
            break
        kept.append(msg)
        used += cost
    return list(reversed(kept))


def _fit_context(chunks: list[ScoredChunk], budget: int) -> list[ScoredChunk]:
    """Add chunks in rank order until the token budget is reached (FR-AUG-03)."""
    kept, used = [], 0
    for sc in chunks:
        if kept and used + sc.chunk.token_count > budget:
            break
        kept.append(sc)
        used += sc.chunk.token_count
    return kept


def _qa_message(ctx: ChatContext, question: str, context: list[ScoredChunk]) -> str:
    excerpts = "\n\n".join(
        f"[{n}] ({sc.chunk.timestamp}) {sc.chunk.text}" for n, sc in enumerate(context, start=1)
    )
    playback = ""
    if ctx.playback_time is not None:
        playback = f"\nThe user is currently at {format_ts(ctx.playback_time)} in the video.\n"
    return QA_TEMPLATE.format(
        title=ctx.index.title or "unknown",
        channel=ctx.index.channel or "unknown",
        length=format_ts(ctx.index.duration_s),
        excerpts=excerpts,
        playback_part=playback,
        question=question,
    )


def _to_messages(history: list[dict]) -> list[BaseMessage]:
    return [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in history
    ]


async def _condense(ctx: ChatContext, history: list[dict], trace: dict) -> tuple[str, CallUsage | None]:
    """Rewrite a follow-up into a standalone query (FR-RET-04). Skipped without history."""
    if not history:
        return ctx.message, None
    convo = "\n".join(f"{m['role'].title()}: {m['content']}" for m in history)
    prompt = CONDENSE_PROMPT.format(history=convo, question=ctx.message)
    t = time.perf_counter()
    result = await ctx.llm.ainvoke([HumanMessage(prompt)], config={**trace, "run_name": "condense"})
    standalone = message_text(result.content).strip()
    usage = call_usage(
        "condense", ctx.provider, ctx.model, result.usage_metadata, prompt, standalone, _ms_since(t)
    )
    return standalone or ctx.message, usage


def _ms_since(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)


# ---------------------------------------------------------------- streaming

async def stream_chat(ctx: ChatContext) -> AsyncIterator[ChatEvent]:
    s = get_settings()
    started = time.perf_counter()
    request_id = uuid.uuid4().hex
    is_summary = bool(_SUMMARY_REQUEST.search(ctx.message))
    local = (
        _excerpt_at(ctx.index, ctx.playback_time)
        if ctx.playback_time is not None and _TIME_LOCAL_REQUEST.search(ctx.message)
        else None
    )
    route = "summary" if is_summary else "time_local" if local else "hybrid"
    trace = {
        "tags": ["yt-chatbot", PROMPT_VERSION],
        "metadata": {
            "request_id": request_id,
            "video_id": ctx.video_id,
            "prompt_version": PROMPT_VERSION,
            "retrieval_strategy": route,
            "provider": ctx.provider,
            "model": ctx.model,
        },
    }

    # Filled in as the request runs; written to the `requests` log at the end,
    # whether it succeeded, failed or was stopped.
    calls: list[CallUsage] = []
    status, error_code = "error", None
    ttft_ms = retrieval_ms = llm_ms = None
    cache_hit = storable = False

    try:
        history = memory.get_history(ctx.video_id, ctx.session_id, limit=s.history_turns * 2)
        history = _fit_history(history, s.history_token_budget)

        query, condense_usage = await _condense(ctx, history, trace)
        if condense_usage:
            calls.append(condense_usage)

        # Semantic cache. "Right now" answers depend on the playhead, so they
        # are never looked up or stored.
        cacheable = ctx.use_cache and local is None
        scope = cache.scope_for(ctx.index, PROMPT_VERSION)
        query_vec = await asyncio.to_thread(embed_query, query)
        hit = await asyncio.to_thread(cache.lookup, ctx.video_id, scope, query, query_vec) if cacheable else None

        if hit:
            cache_hit, low_confidence = True, False
            answer, citations = hit.answer, hit.citations
            retrieval_metrics = {"cache": {"similarity": hit.similarity, "matched_query": hit.matched_query}}
            ttft_ms = _ms_since(started)
            yield ChatEvent("token", {"text": answer})
        else:
            retrieval_metrics: dict = {}
            if is_summary:
                per_chunk = max(s.chunk_max_tokens, 1)
                candidates = spread_chunks(ctx.index, max(s.context_token_budget // per_chunk, 2))
                low_confidence = False
            else:
                t = time.perf_counter()
                result = await asyncio.to_thread(retrieve, ctx.index, query, query_vec)
                retrieval_ms = _ms_since(t)
                candidates, retrieval_metrics = result.chunks, result.metrics
                low_confidence = (
                    result.top_similarity < s.min_similarity
                    and not _VIDEO_DETAILS_REQUEST.search(ctx.message)
                )
                if local:
                    rest = [c for c in candidates if c.chunk.chunk_index != local.chunk.chunk_index]
                    candidates, low_confidence = [local, *rest], False
                    retrieval_metrics["playback_chunk"] = local.chunk.chunk_index

            if low_confidence:
                # FR-GEN-04: say so instead of letting the model improvise.
                route = "not_found"
                answer, citations = NOT_FOUND_ANSWER, []
                ttft_ms = _ms_since(started)
                yield ChatEvent("token", {"text": answer})
            else:
                context = _fit_context(candidates, s.context_token_budget)
                messages = [
                    SystemMessage(SYSTEM_PROMPT),
                    *_to_messages(history),
                    HumanMessage(_qa_message(ctx, query, context)),
                ]
                raw = ""
                streamed = None  # chunks summed: providers send usage as per-chunk deltas
                t = time.perf_counter()
                try:
                    async for part in ctx.llm.astream(messages, config={**trace, "run_name": "qa"}):
                        streamed = part if streamed is None else streamed + part
                        text = message_text(part.content)
                        if text:
                            if ttft_ms is None:
                                ttft_ms = _ms_since(started)
                            raw += text
                            yield ChatEvent("token", {"text": text})
                finally:
                    # Also on Stop: tokens generated so far were still paid for.
                    llm_ms = _ms_since(t)
                    prompt_text = "\n".join(message_text(m.content) for m in messages)
                    calls.append(call_usage(
                        "qa", ctx.provider, ctx.model,
                        streamed.usage_metadata if streamed is not None else None,
                        prompt_text, raw, llm_ms,
                    ))

                if guardrails.leaks_system_prompt(raw):
                    answer, citations = "Sorry, I can't share that.", []
                else:
                    answer, citations = guardrails.apply_citations(raw, context)
                    storable = True

        latency_ms = _ms_since(started)
        yield ChatEvent("citations", {"citations": citations})
        yield ChatEvent(
            "done",
            {
                "request_id": request_id,
                "answer": answer,
                "low_confidence": low_confidence,
                "standalone_query": query if query != ctx.message else None,
                "prompt_version": PROMPT_VERSION,
                "provider": ctx.provider,
                "model": ctx.model,
                "route": route,
                "cache_hit": cache_hit,
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
                "usage": usage_summary(calls),
                "retrieval": retrieval_metrics,
            },
        )
        status = "ok"

        # Only first-turn answers are stored: a follow-up's answer was written
        # with the conversation in view and may not stand on its own.
        if cacheable and storable and not history:
            await asyncio.to_thread(
                cache.save, ctx.video_id, scope, query, query_vec, answer, citations, request_id
            )

        memory.add_message(ctx.video_id, ctx.session_id, "user", ctx.message)
        memory.add_message(
            ctx.video_id, ctx.session_id, "assistant", answer,
            citations=citations, prompt_version=PROMPT_VERSION, latency_ms=latency_ms,
        )
        logger.info(
            "Answered %s/%s in %d ms (route=%s, cache_hit=%s, usage=%s)",
            ctx.video_id, ctx.session_id, latency_ms, route, cache_hit, usage_summary(calls),
        )

    except (asyncio.CancelledError, GeneratorExit):
        status = "stopped"  # the client hit Stop or disconnected
        raise
    except Exception as e:
        logger.exception("Chat failed for %s", ctx.video_id)
        error_code, message = _describe_llm_error(e)
        yield ChatEvent("error", {"code": error_code, "message": message})
    finally:
        log_request(
            request_id=request_id,
            video_id=ctx.video_id,
            session_id=ctx.session_id,
            status=status,
            error_code=error_code,
            route=route,
            provider=ctx.provider,
            model=ctx.model,
            prompt_version=PROMPT_VERSION,
            cache_hit=cache_hit,
            calls=calls,
            latency_ms=_ms_since(started),
            ttft_ms=ttft_ms,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
        )


# Provider SDKs raise different exception types; their messages carry the HTTP
# status or the provider's error name, which is enough to tell the user what to do.
_LLM_ERRORS = [
    (r"\b503\b|UNAVAILABLE|overloaded|high demand", "llm_unavailable",
     "The AI model is overloaded right now. Try again in a minute, or pick another model in Settings."),
    (r"\b429\b|RESOURCE_EXHAUSTED|rate.?limit|quota", "llm_rate_limited",
     "The AI provider's rate limit or free quota was reached. Wait a bit, or use another provider in Settings."),
    (r"\b40[13]\b|API.?key|PERMISSION_DENIED|unauthori[sz]ed|invalid.?key", "llm_auth",
     "The AI provider rejected the API key. Check the key in Settings or the backend .env."),
    (r"\b404\b|NOT_FOUND|not found|no longer available", "llm_model_not_found",
     "That model isn't available. Pick another model in Settings."),
]


def _describe_llm_error(e: Exception) -> tuple[str, str]:
    text = f"{type(e).__name__} {e}"
    for pattern, code, message in _LLM_ERRORS:
        if re.search(pattern, text, re.IGNORECASE):
            return code, message
    return "llm_error", "The answer could not be generated. Please try again."
