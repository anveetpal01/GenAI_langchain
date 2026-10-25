# YouTube Video Chat: Chrome extension

Side-panel chat for the YouTube video you are watching (Manifest V3, Preact + TypeScript + Vite).
Needs the backend running: see [`../backend/README.md`](../backend/README.md).

## Build and load

```bash
cd YoutubeChatBot/extension
npm install
npm run build
```

1. Open `chrome://extensions` and turn on **Developer mode** (top right).
2. Click **Load unpacked** and pick the `extension/dist` folder.
3. Open any YouTube video and click the extension's toolbar icon. The side panel opens.

After a code change: `npm run build`, then click the reload icon on the extension's card in `chrome://extensions`. Reload YouTube tabs that were already open so they get the new content script.

## Develop the UI without reloading the extension

```bash
npm run dev
```

This opens http://localhost:5173/sidepanel/index.html as a normal web page. A fake `chrome` API (`src/dev/chrome-shim.ts`, dev only, never in the build) pretends a demo video is open at 07:00, so the panel talks to the real backend. The settings page is at http://localhost:5173/options/index.html.

## How it fits together

```
src/
├── background/   service worker: side panel only on youtube.com/watch, opens on icon click
├── content/      runs in YouTube pages: current video id (incl. in-page navigation), seek, playback time
├── sidepanel/    the chat UI (App.tsx + components/)
├── options/      settings: backend URL, LLM provider/model/key (BYOK), auto-index, connection test
├── shared/       backend client, SSE reader, storage, message types, theme
└── dev/          chrome API shim for `npm run dev`
```

- The panel asks the content script for the video and time with `chrome.tabs.sendMessage`; the content script announces video changes with `chrome.runtime.sendMessage`.
- A video is indexed on the first question (or as soon as it opens, if enabled in Settings).
- Answers stream over SSE (`fetch` + `src/shared/sse.ts`; `EventSource` can't POST). `[n]` markers become timestamp chips that seek the video.
- Chats, settings and API keys live in `chrome.storage.local` only.
