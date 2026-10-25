import { useCallback, useEffect, useState } from "preact/hooks";
import type { PanelRequest, VideoChangedEvent, VideoInfo } from "../shared/messages";

export interface ActiveVideo extends VideoInfo {
  tabId: number | null;
  /** False when the tab was open before the extension was installed/reloaded. */
  contentScriptReady: boolean;
}

const EMPTY: ActiveVideo = { tabId: null, videoId: null, title: null, contentScriptReady: false };

export function sendToTab<T>(tabId: number, msg: PanelRequest): Promise<T> {
  return chrome.tabs.sendMessage(tabId, msg) as Promise<T>;
}

/** The video in the active tab of this window; follows tab switches and
 *  YouTube's in-page navigation. */
export function useActiveVideo(): ActiveVideo {
  const [active, setActive] = useState<ActiveVideo>(EMPTY);

  const refresh = useCallback(async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tab?.id == null) return setActive(EMPTY);
    try {
      const info = await sendToTab<VideoInfo>(tab.id, { type: "get-video" });
      setActive({ tabId: tab.id, ...info, contentScriptReady: true });
    } catch {
      // No content script in this tab yet: fall back to the URL.
      const url = tab.url ? new URL(tab.url) : null;
      const videoId = url?.pathname === "/watch" ? url.searchParams.get("v") : null;
      setActive({ tabId: tab.id, videoId, title: null, contentScriptReady: false });
    }
  }, []);

  useEffect(() => {
    void refresh();
    const onMessage = (msg: VideoChangedEvent, sender: chrome.runtime.MessageSender) => {
      if (msg?.type === "video-changed" && sender.tab?.active) void refresh();
    };
    const onActivated = () => void refresh();
    const onUpdated = (_id: number, change: { url?: string }, tab: chrome.tabs.Tab) => {
      if (change.url && tab.active) void refresh();
    };
    chrome.runtime.onMessage.addListener(onMessage);
    chrome.tabs.onActivated.addListener(onActivated);
    chrome.tabs.onUpdated.addListener(onUpdated);
    return () => {
      chrome.runtime.onMessage.removeListener(onMessage);
      chrome.tabs.onActivated.removeListener(onActivated);
      chrome.tabs.onUpdated.removeListener(onUpdated);
    };
  }, [refresh]);

  return active;
}
