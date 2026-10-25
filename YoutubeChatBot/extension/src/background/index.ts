// Service worker: the side panel exists only on youtube.com/watch pages
// (SRS FR-UI-01) and opens when the toolbar icon is clicked.

const WATCH_URL = /^https:\/\/(www\.|m\.)?youtube\.com\/watch\?/;

chrome.runtime.onInstalled.addListener(() => {
  void chrome.sidePanel.setPanelBehavior({ openPanelOnActionClick: true });
});

async function updatePanel(tabId: number, url: string | undefined): Promise<void> {
  await chrome.sidePanel.setOptions({
    tabId,
    path: "sidepanel/index.html",
    enabled: WATCH_URL.test(url ?? ""),
  });
}

chrome.tabs.onUpdated.addListener((tabId, change, tab) => {
  if (change.url || change.status === "complete") void updatePanel(tabId, tab.url);
});
