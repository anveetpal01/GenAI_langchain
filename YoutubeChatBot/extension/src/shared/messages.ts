// Messages between the side panel and the content script. Types only, so the
// content script can import them without pulling in a shared chunk.

export interface VideoInfo {
  videoId: string | null;
  title: string | null;
}

/** Side panel -> content script (chrome.tabs.sendMessage). */
export type PanelRequest =
  | { type: "get-video" }
  | { type: "get-time" }
  | { type: "seek"; seconds: number }
  | { type: "get-transcript"; videoId: string };

export interface TimeResponse {
  seconds: number | null;
}

/** Captions fetched in the user's browser and sent to the backend for
 *  indexing: YouTube blocks transcript requests from cloud servers. Field
 *  names match the backend's IndexRequest. */
export interface TranscriptPayload {
  captions: { text: string; start_s: number; end_s: number }[];
  language_code: string;
  is_generated: boolean;
  title: string | null;
  channel: string | null;
}

export type TranscriptResponse =
  | { ok: true; transcript: TranscriptPayload }
  | { ok: false; code: string; message: string };

/** Content script -> extension (chrome.runtime.sendMessage). */
export interface VideoChangedEvent extends VideoInfo {
  type: "video-changed";
}
