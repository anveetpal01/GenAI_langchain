"""Timestamp-aware chunking of caption segments (SRS FR-IDX-03, FR-IDX-04).

Captions are merged, in order, into windows of at most `max_tokens` tokens.
Consecutive chunks share roughly `overlap_tokens` tokens of trailing captions
so that an idea split across a boundary is still retrievable. Every chunk keeps
the start time of its first caption and the end time of its last one, so an
answer can cite `[mm:ss]` and the extension can seek the video there.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import tiktoken

from app.transcript import Caption, VideoTranscript

logger = logging.getLogger(__name__)

# bge-small-en-v1.5 embeds at most 512 (WordPiece) tokens; anything longer is
# silently truncated. 400 cl100k tokens leaves headroom for that difference.
DEFAULT_MAX_TOKENS = 400
DEFAULT_OVERLAP_TOKENS = 60  # ~15 %

TokenCounter = Callable[[str], int]

_encoding = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding.encode(text))


@dataclass(frozen=True)
class Chunk:
    video_id: str
    chunk_index: int
    text: str
    start_s: float
    end_s: float
    token_count: int

    @property
    def timestamp(self) -> str:
        """`mm:ss-mm:ss` (or `h:mm:ss`) label used in prompts and citations."""
        return f"{format_ts(self.start_s)}-{format_ts(self.end_s)}"


def format_ts(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _make_chunk(
    video_id: str, index: int, captions: Sequence[Caption], counter: TokenCounter
) -> Chunk:
    text = " ".join(c.text for c in captions)
    return Chunk(
        video_id=video_id,
        chunk_index=index,
        text=text,
        start_s=captions[0].start_s,
        # Auto-generated captions overlap in time, so the last caption does
        # not always end last.
        end_s=max(c.end_s for c in captions),
        token_count=counter(text),
    )


def chunk_transcript(
    transcript: VideoTranscript,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    counter: TokenCounter = count_tokens,
) -> list[Chunk]:
    """Split a transcript into overlapping, timestamped chunks."""
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be smaller than max_tokens")

    captions = transcript.captions
    # Count each caption once. The leading space mirrors how captions are
    # joined, so the sum stays close to counter(chunk.text).
    sizes = [counter(" " + c.text) for c in captions]

    chunks: list[Chunk] = []
    start = 0
    while start < len(captions):
        # Grow the window until the next caption would overflow it.
        # A single caption larger than max_tokens becomes its own chunk.
        end, size = start, 0
        while end < len(captions) and (end == start or size + sizes[end] <= max_tokens):
            size += sizes[end]
            end += 1

        chunks.append(_make_chunk(transcript.video_id, len(chunks), captions[start:end], counter))
        if end == len(captions):
            break

        # Step back over trailing captions to build the overlap, but always
        # move forward by at least one caption so the loop terminates.
        next_start, overlap = end, 0
        while next_start - 1 > start and overlap + sizes[next_start - 1] <= overlap_tokens:
            next_start -= 1
            overlap += sizes[next_start]
        start = next_start

    logger.info(
        "Split %d captions of %s into %d chunks (max=%d, overlap=%d tokens)",
        len(captions), transcript.video_id, len(chunks), max_tokens, overlap_tokens,
    )
    return chunks


if __name__ == "__main__":
    from app.transcript import get_transcript

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = get_transcript("IYFzj-7wGA0")
    chunks = chunk_transcript(result)

    for chunk in chunks:
        print(
            f"#{chunk.chunk_index:<3} [{chunk.timestamp}] {chunk.token_count:>3} tok | "
            f"{chunk.text[:60]}...{chunk.text[-30:]}"
        )
