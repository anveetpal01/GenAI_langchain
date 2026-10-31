// Runs inside youtube.com pages (SRS FR-UI-02, FR-UI-05).
// - tells the extension which video is open, including YouTube's in-page
//   (SPA) navigation, where the page never reloads
// - answers "which video?" / "what time is it?" and seeks the player
// - fetches the transcript from the user's own browser (YouTube blocks
//   transcript requests from cloud servers like AWS)
//
// Must not import runtime code (see vite.config.ts); `import type` is fine.
import type {
  PanelRequest,
  TimeResponse,
  TranscriptPayload,
  TranscriptResponse,
  VideoChangedEvent,
  VideoInfo,
} from "../shared/messages";

function currentVideoId(): string | null {
  const url = new URL(location.href);
  return url.pathname === "/watch" ? url.searchParams.get("v") : null;
}

function currentTitle(): string | null {
  const heading = document.querySelector("h1.ytd-watch-metadata")?.textContent?.trim();
  return heading || document.title.replace(/ - YouTube$/, "") || null;
}

function player(): HTMLVideoElement | null {
  return document.querySelector<HTMLVideoElement>("video.html5-main-video") ?? document.querySelector("video");
}

function videoInfo(): VideoInfo {
  const videoId = currentVideoId();
  return { videoId, title: videoId ? currentTitle() : null };
}

let announced: string | null | undefined;

function announce(): void {
  const info = videoInfo();
  if (info.videoId === announced) return;
  announced = info.videoId;
  const event: VideoChangedEvent = { type: "video-changed", ...info };
  // Rejects when no extension page is listening (panel closed): that's fine.
  chrome.runtime.sendMessage(event).catch(() => {});
}

// Fired by YouTube after every in-app navigation.
document.addEventListener("yt-navigate-finish", announce);
announce();

// ---------------------------------------------------------------- transcript

// Same steps as the backend's youtube-transcript-api: the watch page gives the
// InnerTube API key, the player API lists caption tracks, and the json3 format
// of a track has the timed lines. The ANDROID client context mirrors that
// library; if YouTube changes it, both break the same way.
const INNERTUBE_CONTEXT = { client: { clientName: "ANDROID", clientVersion: "20.10.38" } };
const PREFERRED_LANGUAGES = ["en", "en-US", "en-GB"];

interface CaptionTrack {
  baseUrl: string;
  languageCode: string;
  kind?: string;
}

class TranscriptError extends Error {
  constructor(
    public code: string,
    message: string,
  ) {
    super(message);
  }
}

/** Manual English, then auto-generated English, then any track (as the backend does). */
function pickTrack(tracks: CaptionTrack[]): CaptionTrack | null {
  const manual = tracks.filter((t) => t.kind !== "asr");
  const generated = tracks.filter((t) => t.kind === "asr");
  for (const group of [manual, generated]) {
    for (const code of PREFERRED_LANGUAGES) {
      const track = group.find((t) => t.languageCode === code);
      if (track) return track;
    }
  }
  return tracks[0] ?? null;
}

async function fetchTranscript(videoId: string): Promise<TranscriptPayload> {
  const html = await (await fetch(`/watch?v=${videoId}`)).text();
  const apiKey = /"INNERTUBE_API_KEY":\s*"([\w-]+)"/.exec(html)?.[1];
  if (!apiKey) throw new TranscriptError("youtube_unparsable", "Couldn't read this YouTube page. Reload it and try again.");

  const player = await (
    await fetch(`/youtubei/v1/player?key=${apiKey}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ context: INNERTUBE_CONTEXT, videoId }),
    })
  ).json();

  const status = player?.playabilityStatus?.status;
  if (status && status !== "OK") {
    throw new TranscriptError("video_unplayable", player.playabilityStatus.reason ?? "This video can't be played.");
  }
  const tracks: CaptionTrack[] = player?.captions?.playerCaptionsTracklistRenderer?.captionTracks ?? [];
  const track = pickTrack(tracks);
  if (!track) throw new TranscriptError("captions_disabled", "This video has no captions, so it can't be chatted with.");

  const url = new URL(track.baseUrl, location.origin);
  url.searchParams.set("fmt", "json3");
  const json = await (await fetch(url)).json();

  const captions = (json.events ?? [])
    .filter((e: { segs?: unknown }) => e.segs)
    .map((e: { tStartMs: number; dDurationMs?: number; segs: { utf8?: string }[] }) => ({
      text: e.segs.map((s) => s.utf8 ?? "").join("").replace(/\s+/g, " ").trim(),
      start_s: e.tStartMs / 1000,
      end_s: (e.tStartMs + (e.dDurationMs ?? 0)) / 1000,
    }))
    .filter((c: { text: string }) => c.text);

  return {
    captions,
    language_code: track.languageCode,
    is_generated: track.kind === "asr",
    title: player?.videoDetails?.title ?? null,
    channel: player?.videoDetails?.author ?? null,
  };
}

// ---------------------------------------------------------------- messages

chrome.runtime.onMessage.addListener((msg: PanelRequest, _sender, sendResponse) => {
  if (msg.type === "get-transcript") {
    fetchTranscript(msg.videoId)
      .then((transcript) => sendResponse({ ok: true, transcript } satisfies TranscriptResponse))
      .catch((err) => {
        const code = err instanceof TranscriptError ? err.code : "transcript_fetch_failed";
        const message = err instanceof TranscriptError ? err.message : "Couldn't load this video's captions. Try again.";
        sendResponse({ ok: false, code, message } satisfies TranscriptResponse);
      });
    return true; // keep the channel open for the async response
  }

  switch (msg.type) {
    case "get-video":
      sendResponse(videoInfo());
      break;
    case "get-time": {
      const response: TimeResponse = { seconds: player()?.currentTime ?? null };
      sendResponse(response);
      break;
    }
    case "seek": {
      const video = player();
      if (video) {
        video.currentTime = msg.seconds;
        void video.play().catch(() => {});
      }
      sendResponse({ ok: Boolean(video) });
      break;
    }
  }
});
