"""Fetch YouTube captions for a video, with timestamps and typed errors.

Language preference (SRS FR-IDX-01):
    manual English -> auto-generated English -> any available language.
Every failure is raised as `TranscriptError` with a stable `code` (for the
API / UI) and a human-readable `message` (SRS FR-IDX-02).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    InvalidVideoId,
    NoTranscriptFound,
    RequestBlocked,
    Transcript,
    TranscriptList,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeTranscriptApi,
)

logger = logging.getLogger(__name__)

PREFERRED_LANGUAGES = ("en", "en-US", "en-GB")


class TranscriptError(Exception):
    """Raised when captions for a video cannot be fetched."""

    def __init__(self, code: str, message: str, video_id: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.video_id = video_id

    def __str__(self) -> str:
        return f"[{self.code}] {self.message} (video_id={self.video_id})"


@dataclass(frozen=True)
class Caption:
    text: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class VideoTranscript:
    video_id: str
    language_code: str
    is_generated: bool
    captions: list[Caption] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return self.captions[-1].end_s if self.captions else 0.0


def _pick_transcript(transcript_list: TranscriptList) -> Transcript:
    """Manual English first, then auto-generated English, then anything."""
    for finder in (
        transcript_list.find_manually_created_transcript,
        transcript_list.find_generated_transcript,
    ):
        try:
            return finder(PREFERRED_LANGUAGES)
        except NoTranscriptFound:
            continue

    # No English at all: fall back to the first available transcript.
    for transcript in transcript_list:
        return transcript

    raise NoTranscriptFound(transcript_list.video_id, PREFERRED_LANGUAGES, transcript_list)


def _clean_text(text: str) -> str:
    # Captions often contain line breaks and double spaces.
    return " ".join(text.split())


def get_transcript(video_id: str) -> VideoTranscript:
    """Return the captions of `video_id` with start/end times in seconds.

    Raises:
        TranscriptError: if captions are unavailable for any reason.
    """
    api = YouTubeTranscriptApi()
    try:
        transcript = _pick_transcript(api.list(video_id))
        fetched = transcript.fetch()

    # Order matters: subclasses before their parents
    # (IpBlocked < RequestBlocked < CouldNotRetrieveTranscript).
    except TranscriptsDisabled as e:
        raise TranscriptError(
            "captions_disabled", "Captions are turned off for this video.", video_id
        ) from e
    except NoTranscriptFound as e:
        raise TranscriptError(
            "no_captions", "This video has no captions in any language.", video_id
        ) from e
    except InvalidVideoId as e:
        raise TranscriptError(
            "invalid_video_id",
            "Invalid video ID. Pass only the 11-character ID, not the full URL.",
            video_id,
        ) from e
    except VideoUnavailable as e:
        raise TranscriptError(
            "video_unavailable", "This video is private, deleted or does not exist.", video_id
        ) from e
    except VideoUnplayable as e:
        raise TranscriptError(
            "video_unplayable",
            "This video can't be played (age-restricted or region-locked).",
            video_id,
        ) from e
    except RequestBlocked as e:
        raise TranscriptError(
            "blocked", "YouTube is blocking requests from this network. Try again later.", video_id
        ) from e
    except CouldNotRetrieveTranscript as e:
        raise TranscriptError(
            "unknown", "Could not fetch the transcript for this video.", video_id
        ) from e

    captions = [
        Caption(
            text=_clean_text(snippet.text),
            start_s=round(snippet.start, 2),
            end_s=round(snippet.start + snippet.duration, 2),
        )
        for snippet in fetched
        if snippet.text.strip()
    ]

    logger.info(
        "Fetched %d captions for %s (lang=%s, generated=%s)",
        len(captions), video_id, transcript.language_code, transcript.is_generated,
    )
    return VideoTranscript(
        video_id=video_id,
        language_code=transcript.language_code,
        is_generated=transcript.is_generated,
        captions=captions,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    test_ids = [
        "IYFzj-7wGA0",                                  # valid video
        "xxxxxxxxxxx",                                  # does not exist
        "https://www.youtube.com/watch?v=IYFzj-7wGA0",  # URL instead of ID
    ]
    for vid in test_ids:
        print(f"\n=== {vid} ===")
        try:
            result = get_transcript(vid)
        except TranscriptError as err:
            print(f"ERROR code={err.code!r}: {err.message}")
            continue
        print(
            f"lang={result.language_code} generated={result.is_generated} "
            f"captions={len(result.captions)} duration={result.duration_s:.0f}s"
        )
        for caption in result.captions[:3]:
            print(" ", caption)
