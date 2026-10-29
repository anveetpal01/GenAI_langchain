// Everything persisted lives in chrome.storage.local (SRS FR-UI-06, NFR-03):
// it never syncs to Google and never leaves this browser profile.

export interface Settings {
  backendUrl: string;
  /** Empty = use the backend's .env default. */
  provider: string;
  model: string;
  apiKey: string;
  /** Sent as X-API-Key; must match API_KEY in backend/.env. */
  backendKey: string;
  /** Index as soon as a video opens, instead of on the first question (SRS D-11). */
  autoIndex: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  backendUrl: "http://localhost:8000",
  provider: "",
  model: "",
  apiKey: "",
  backendKey: "",
  autoIndex: false,
};

export interface Citation {
  n: number;
  chunk_index: number;
  start_s: number;
  end_s: number;
  timestamp: string;
  text: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "error";
  content: string;
  citations?: Citation[];
  /** For role "error": the ApiError code, e.g. "backend_unreachable". */
  code?: string;
}

export interface VideoChat {
  sessionId: string;
  messages: ChatMessage[];
}

export async function loadSettings(): Promise<Settings> {
  const { settings } = await chrome.storage.local.get("settings");
  return { ...DEFAULT_SETTINGS, ...(settings as Partial<Settings> | undefined) };
}

export async function saveSettings(settings: Settings): Promise<void> {
  await chrome.storage.local.set({ settings });
}

export function newId(): string {
  return crypto.randomUUID().replace(/-/g, "").slice(0, 24);
}

const chatKey = (videoId: string) => `chat:${videoId}`;

export async function loadChat(videoId: string): Promise<VideoChat> {
  const key = chatKey(videoId);
  const stored = (await chrome.storage.local.get(key))[key] as VideoChat | undefined;
  return stored ?? { sessionId: newId(), messages: [] };
}

export async function saveChat(videoId: string, chat: VideoChat): Promise<void> {
  await chrome.storage.local.set({ [chatKey(videoId)]: chat });
}
