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
  | { type: "seek"; seconds: number };

export interface TimeResponse {
  seconds: number | null;
}

/** Content script -> extension (chrome.runtime.sendMessage). */
export interface VideoChangedEvent extends VideoInfo {
  type: "video-changed";
}
