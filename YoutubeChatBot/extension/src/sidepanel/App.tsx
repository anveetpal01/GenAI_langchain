import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { ApiError, BackendClient, type VideoStatus } from "../shared/api";
import type { TimeResponse } from "../shared/messages";
import {
  type ChatMessage,
  type Settings,
  type VideoChat,
  loadChat,
  loadSettings,
  newId,
  saveChat,
} from "../shared/storage";
import { Composer, QUICK_ACTIONS } from "./components/Composer";
import { Header } from "./components/Header";
import { MessageList } from "./components/MessageList";
import { sendToTab, useActiveVideo } from "./useActiveVideo";

interface ChatState {
  videoId: string;
  chat: VideoChat;
}

function errorMessage(err: unknown): ChatMessage {
  if (err instanceof ApiError) return { id: newId(), role: "error", content: err.message, code: err.code };
  return { id: newId(), role: "error", content: "Something went wrong. Please try again." };
}

export function App() {
  const video = useActiveVideo();
  const [settings, setSettings] = useState<Settings | null>(null);
  const [state, setState] = useState<ChatState | null>(null);
  const [status, setStatus] = useState<VideoStatus | null>(null);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  const videoId = video.videoId;
  const client = settings ? new BackendClient(settings) : null;

  // Settings: load once, then follow changes made on the options page.
  useEffect(() => {
    void loadSettings().then(setSettings);
    const onChange = (changes: Record<string, chrome.storage.StorageChange>) => {
      if (changes.settings) void loadSettings().then(setSettings);
    };
    chrome.storage.onChanged?.addListener(onChange);
    return () => chrome.storage.onChanged?.removeListener(onChange);
  }, []);

  // A different video: stop any answer in flight, load its chat and status.
  useEffect(() => {
    abort.current?.abort();
    setState(null);
    setStatus(null);
    setNotice(null);
    if (!videoId || !settings) return;

    let cancelled = false;
    void loadChat(videoId).then((chat) => !cancelled && setState({ videoId, chat }));
    const api = new BackendClient(settings);
    const ready = settings.autoIndex
      ? api.ensureIndexed(videoId, (s) => !cancelled && setStatus(s))
      : api.status(videoId).then((s) => !cancelled && setStatus(s));
    ready.catch(() => {}); // surfaced when the user asks a question
    return () => {
      cancelled = true;
    };
  }, [videoId, settings]);

  // Persist the chat whenever it settles (not on every streamed token).
  useEffect(() => {
    if (state && !streamingId) void saveChat(state.videoId, state.chat);
  }, [state, streamingId]);

  /** Update messages of `forVideo` only; ignores updates after a video switch. */
  const updateMessages = useCallback((forVideo: string, fn: (m: ChatMessage[]) => ChatMessage[]) => {
    setState((s) => (s && s.videoId === forVideo ? { ...s, chat: { ...s.chat, messages: fn(s.chat.messages) } } : s));
  }, []);

  const send = async (text: string) => {
    if (!videoId || !client || !state || streamingId) return;
    const forVideo = videoId;
    const answerId = newId();
    const controller = new AbortController();
    abort.current = controller;
    setStreamingId(answerId);
    updateMessages(forVideo, (m) => [
      ...m,
      { id: newId(), role: "user", content: text },
      { id: answerId, role: "assistant", content: "" },
    ]);
    const patchAnswer = (patch: Partial<ChatMessage> | ((a: ChatMessage) => Partial<ChatMessage>)) =>
      updateMessages(forVideo, (m) =>
        m.map((msg) => (msg.id === answerId ? { ...msg, ...(typeof patch === "function" ? patch(msg) : patch) } : msg)),
      );

    try {
      await client.ensureIndexed(forVideo, setStatus);

      let playbackTime: number | null = null;
      if (video.tabId != null && video.contentScriptReady) {
        playbackTime = await sendToTab<TimeResponse>(video.tabId, { type: "get-time" })
          .then((r) => r.seconds)
          .catch(() => null);
      }

      await client.chat(
        forVideo,
        { session_id: state.chat.sessionId, message: text, playback_time: playbackTime },
        {
          onToken: (t) => patchAnswer((a) => ({ content: a.content + t })),
          onCitations: (citations) => patchAnswer({ citations }),
          // The final answer has invalid citation markers removed.
          onDone: (done) => patchAnswer({ content: done.answer }),
        },
        controller.signal,
      );
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        patchAnswer((a) => ({ content: a.content ? `${a.content} …(stopped)` : "(stopped)" }));
      } else {
        // Replace the empty/partial answer with the error.
        updateMessages(forVideo, (m) => [...m.filter((msg) => msg.id !== answerId || msg.content), errorMessage(err)]);
      }
    } finally {
      if (abort.current === controller) abort.current = null;
      setStreamingId(null);
    }
  };

  const stop = () => abort.current?.abort();

  const seek = async (seconds: number) => {
    if (video.tabId == null) return;
    try {
      await sendToTab(video.tabId, { type: "seek", seconds });
      setNotice(null);
    } catch {
      setNotice("Reload the YouTube tab once so the extension can control the player.");
    }
  };

  const clearChat = async () => {
    if (!videoId || !state) return;
    stop();
    const old = state.chat.sessionId;
    setState({ videoId, chat: { sessionId: newId(), messages: [] } });
    await client?.deleteSession(videoId, old).catch(() => {});
  };

  const reindex = async () => {
    if (!videoId || !client) return;
    try {
      await client.ensureIndexed(videoId, setStatus, { force: true });
    } catch (err) {
      updateMessages(videoId, (m) => [...m, errorMessage(err)]);
    }
  };

  const openSettings = () => void chrome.runtime.openOptionsPage();

  if (!videoId) {
    return (
      <main class="empty">
        <p class="empty-title">Open a YouTube video</p>
        <p class="muted">Then ask anything about it here. Answers link to the exact moment in the video.</p>
      </main>
    );
  }

  const messages = state?.chat.messages ?? [];
  const busy = streamingId !== null || status?.state === "indexing";

  return (
    <main class="panel">
      <Header
        title={video.title}
        status={status}
        busy={busy}
        onReindex={reindex}
        onClear={clearChat}
        onSettings={openSettings}
      />
      {notice && (
        <p class="notice" role="status">
          {notice}
        </p>
      )}
      {messages.length === 0 ? (
        <div class="welcome">
          <p class="empty-title">Ask about this video</p>
          <p class="muted">For example:</p>
          <ul class="examples">
            {QUICK_ACTIONS.map((a) => (
              <li key={a.label}>
                <button type="button" class="link-btn" disabled={!state} onClick={() => send(a.message)}>
                  {a.message}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : (
        <MessageList messages={messages} streamingId={streamingId} onSeek={seek} onOpenSettings={openSettings} />
      )}
      <Composer disabled={!state} streaming={streamingId !== null} onSend={send} onStop={stop} />
    </main>
  );
}
