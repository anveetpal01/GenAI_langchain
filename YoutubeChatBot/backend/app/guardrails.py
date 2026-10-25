"""Input and output guardrails (SRS FR-GEN-05, FR-GEN-06, FR-GEN-03)."""

from __future__ import annotations

import re
import threading
import time
from collections import defaultdict, deque

from app.config import get_settings
from app.prompts import SYSTEM_PROMPT
from app.retrieval import ScoredChunk


class GuardrailError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


# ---------------------------------------------------------------- input

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Kept narrow on purpose: a question *about* prompts in a video is legitimate.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions|rules|prompts?)",
        r"\b(reveal|print|show|repeat)\s+(me\s+)?(your|the)\s+(system\s+prompt|instructions|rules)",
        r"\byou\s+are\s+now\s+(in\s+)?(developer|dan|jailbreak)\b",
    )
]


def clean_message(text: str) -> str:
    text = _CONTROL_CHARS.sub("", text).strip()
    if not text:
        raise GuardrailError("empty_message", "Message is empty.")
    limit = get_settings().max_message_chars
    if len(text) > limit:
        raise GuardrailError("message_too_long", f"Message is longer than {limit} characters.")
    if any(p.search(text) for p in _INJECTION_PATTERNS):
        raise GuardrailError(
            "unsafe_message", "That looks like an attempt to change my instructions. Ask about the video instead."
        )
    return text


class RateLimiter:
    """Sliding-window limit per key, in memory (single-user local backend)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        limit = get_settings().rate_limit_per_min
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= limit:
                raise GuardrailError(
                    "rate_limited", f"Too many messages. Limit is {limit} per minute.", status=429
                )
            hits.append(now)


rate_limiter = RateLimiter()


# ---------------------------------------------------------------- output

_CITATION_GROUP = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def apply_citations(answer: str, context: list[ScoredChunk]) -> tuple[str, list[dict]]:
    """Normalise `[1, 2]` to `[1][2]`, drop markers that point at no excerpt,
    and return the citations actually used, in order of first appearance."""
    used: list[int] = []

    def replace(match: re.Match) -> str:
        valid = []
        for raw in match.group(1).split(","):
            n = int(raw)
            if 1 <= n <= len(context):
                valid.append(n)
                if n not in used:
                    used.append(n)
        return "".join(f"[{n}]" for n in valid)

    cleaned = _CITATION_GROUP.sub(replace, answer)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)  # space left by a dropped marker

    citations = []
    for n in used:
        chunk = context[n - 1].chunk
        citations.append(
            {
                "n": n,
                "chunk_index": chunk.chunk_index,
                "start_s": chunk.start_s,
                "end_s": chunk.end_s,
                "timestamp": chunk.timestamp,
                "text": chunk.text[:300],
            }
        )
    return cleaned, citations


_PROMPT_LINES = [line.strip() for line in SYSTEM_PROMPT.splitlines() if len(line.strip()) > 40]


def leaks_system_prompt(answer: str) -> bool:
    lowered = answer.lower()
    return any(line.lower() in lowered for line in _PROMPT_LINES)
