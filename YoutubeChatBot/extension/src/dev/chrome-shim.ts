// Development only: lets `npm run dev` open the side panel and options pages
// as normal web pages (http://localhost:5173) with a fake `chrome` API, so the
// UI can be worked on against the real backend without reloading the
// extension. Never included in `npm run build` (guarded by import.meta.env.DEV).

const DEMO_VIDEO = { videoId: "IYFzj-7wGA0", title: "The Only 17 Minutes You Need To Master Your Mind" };
let playbackSeconds = 420;

const storage = {
  async get(key: string) {
    const raw = localStorage.getItem(`shim:${key}`);
    return raw ? { [key]: JSON.parse(raw) } : {};
  },
  async set(items: Record<string, unknown>) {
    for (const [k, v] of Object.entries(items)) localStorage.setItem(`shim:${k}`, JSON.stringify(v));
  },
};

const noopEvent = { addListener() {}, removeListener() {} };

const shim = {
  storage: { local: storage },
  runtime: {
    id: "dev-shim",
    onMessage: noopEvent,
    openOptionsPage: () => window.open("/options/index.html", "_blank"),
  },
  tabs: {
    onActivated: noopEvent,
    onUpdated: noopEvent,
    async query() {
      return [{ id: 1, url: `https://www.youtube.com/watch?v=${DEMO_VIDEO.videoId}` }];
    },
    async sendMessage(_tabId: number, msg: { type: string; seconds?: number }) {
      if (msg.type === "get-video") return DEMO_VIDEO;
      if (msg.type === "get-time") return { seconds: playbackSeconds };
      if (msg.type === "seek") {
        playbackSeconds = msg.seconds ?? 0;
        console.info(`[chrome-shim] seek to ${playbackSeconds}s`);
        return { ok: true };
      }
      return undefined;
    },
  },
};

if (!(globalThis as { chrome?: { runtime?: { id?: string } } }).chrome?.runtime?.id) {
  (globalThis as unknown as { chrome: unknown }).chrome = shim;
}

export {};
