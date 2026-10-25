// Runs inside youtube.com pages (SRS FR-UI-02, FR-UI-05).
// - tells the extension which video is open, including YouTube's in-page
//   (SPA) navigation, where the page never reloads
// - answers "which video?" / "what time is it?" and seeks the player
//
// Must not import runtime code (see vite.config.ts); `import type` is fine.
import type { PanelRequest, TimeResponse, VideoChangedEvent, VideoInfo } from "../shared/messages";

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

chrome.runtime.onMessage.addListener((msg: PanelRequest, _sender, sendResponse) => {
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
