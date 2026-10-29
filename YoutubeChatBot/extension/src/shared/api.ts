// Typed client for the FastAPI backend (backend/README.md).
import { readSse } from "./sse";
import type { Citation, Settings } from "./storage";

export type IndexState = "not_indexed" | "indexing" | "ready" | "failed";

export interface VideoStatus {
  video_id: string;
  state: IndexState;
  chunks: number | null;
  language: string | null;
  title: string | null;
  error: { code: string; message: string } | null;
}

export interface ReadyStatus {
  ready: boolean;
  checks: Record<string, boolean>;
  llm: { provider: string; model: string };
}

export interface DoneEvent {
  answer: string;
  low_confidence: boolean;
  latency_ms: number;
  provider: string;
  model: string;
}

/** Error with a stable machine-readable code (backend RFC 9457 `code`). */
export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status = 0,
  ) {
    super(message);
  }
}

export class BackendClient {
  constructor(private settings: Settings) {}

  private url(path: string): string {
    return this.settings.backendUrl.replace(/\/+$/, "") + path;
  }

  private async request(path: string, init: RequestInit = {}): Promise<Response> {
    let res: Response;
    try {
      const headers = new Headers(init.headers);
      if (this.settings.backendKey) headers.set("X-API-Key", this.settings.backendKey);
      res = await fetch(this.url(path), { ...init, headers });
    } catch (err) {
      if ((err as Error).name === "AbortError") throw err;
      throw new ApiError(
        "backend_unreachable",
        `Can't reach the backend at ${this.settings.backendUrl}. Is it running?`,
      );
    }
    if (!res.ok && res.status !== 503) {
      let code = "http_error";
      let detail = `Request failed (${res.status}).`;
      try {
        const problem = await res.json();
        code = problem.code ?? code;
        detail = problem.detail ?? detail;
      } catch {
        /* not JSON */
      }
      throw new ApiError(code, detail, res.status);
    }
    return res;
  }

  async ready(): Promise<ReadyStatus> {
    return (await this.request("/readyz")).json();
  }

  async status(videoId: string): Promise<VideoStatus> {
    return (await this.request(`/v1/videos/${videoId}/status`)).json();
  }

  async startIndex(videoId: string, force = false): Promise<VideoStatus> {
    const res = await this.request(`/v1/videos/${videoId}/index?force=${force}`, { method: "POST" });
    return res.json();
  }

  /** Start indexing if needed and wait until the video is ready. */
  async ensureIndexed(
    videoId: string,
    onStatus: (s: VideoStatus) => void,
    { force = false, timeoutMs = 180_000 } = {},
  ): Promise<VideoStatus> {
    let status = force ? await this.startIndex(videoId, true) : await this.status(videoId);
    if (!force && status.state !== "ready" && status.state !== "indexing") {
      status = await this.startIndex(videoId);
    }
    const deadline = Date.now() + timeoutMs;
    while (status.state === "indexing") {
      onStatus(status);
      if (Date.now() > deadline) throw new ApiError("index_timeout", "Indexing is taking too long.");
      await new Promise((r) => setTimeout(r, 1500));
      status = await this.status(videoId);
    }
    onStatus(status);
    if (status.state === "failed") {
      throw new ApiError(status.error?.code ?? "index_failed", status.error?.message ?? "Indexing failed.");
    }
    return status;
  }

  async deleteSession(videoId: string, sessionId: string): Promise<void> {
    await this.request(`/v1/videos/${videoId}/sessions/${sessionId}`, { method: "DELETE" });
  }

  /** Stream an answer. Resolves when the stream ends; abort via `signal` (Stop button). */
  async chat(
    videoId: string,
    body: { session_id: string; message: string; playback_time?: number | null },
    handlers: {
      onToken: (text: string) => void;
      onCitations: (citations: Citation[]) => void;
      onDone: (done: DoneEvent) => void;
    },
    signal: AbortSignal,
  ): Promise<void> {
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    // BYOK overrides (SRS FR-API-08); empty fields fall back to the backend's .env.
    if (this.settings.provider) headers["X-LLM-Provider"] = this.settings.provider;
    if (this.settings.model) headers["X-LLM-Model"] = this.settings.model;
    if (this.settings.apiKey) headers["X-LLM-API-Key"] = this.settings.apiKey;

    const res = await this.request(`/v1/videos/${videoId}/chat`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal,
    });
    if (!res.body) throw new ApiError("empty_response", "The backend sent no answer.");

    for await (const { event, data } of readSse(res.body)) {
      const payload = JSON.parse(data);
      if (event === "token") handlers.onToken(payload.text);
      else if (event === "citations") handlers.onCitations(payload.citations);
      else if (event === "done") handlers.onDone(payload);
      else if (event === "error") throw new ApiError(payload.code, payload.message);
    }
  }
}
