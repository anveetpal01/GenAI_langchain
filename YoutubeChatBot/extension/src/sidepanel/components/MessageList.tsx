import { useEffect, useRef } from "preact/hooks";
import type { ChatMessage } from "../../shared/storage";
import { AnswerText } from "./AnswerText";

// Errors the user can fix on the options page.
const SETTINGS_FIXABLE = new Set([
  "backend_unreachable",
  "missing_api_key",
  "llm_unavailable",
  "llm_rate_limited",
  "llm_auth",
  "llm_model_not_found",
  "unknown_provider",
]);

interface Props {
  messages: ChatMessage[];
  streamingId: string | null;
  onSeek: (seconds: number) => void;
  onOpenSettings: () => void;
}

export function MessageList({ messages, streamingId, onSeek, onOpenSettings }: Props) {
  const end = useRef<HTMLDivElement>(null);

  // Keep the newest text in view while it streams in.
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  return (
    <div class="messages" role="log" aria-live="polite" aria-label="Conversation">
      {messages.map((m) => {
        if (m.role === "user") {
          return (
            <div key={m.id} class="msg msg-user">
              {m.content}
            </div>
          );
        }
        if (m.role === "error") {
          return (
            <div key={m.id} class="msg msg-error" role="alert">
              {m.content}
              {m.code && SETTINGS_FIXABLE.has(m.code) && (
                <button type="button" class="link-btn" onClick={onOpenSettings}>
                  Open settings
                </button>
              )}
            </div>
          );
        }
        const thinking = m.id === streamingId && !m.content;
        return (
          <div key={m.id} class="msg msg-assistant">
            {thinking ? (
              <span class="typing" aria-label="Thinking">
                <span />
                <span />
                <span />
              </span>
            ) : (
              <AnswerText text={m.content} citations={m.citations} onSeek={onSeek} />
            )}
          </div>
        );
      })}
      <div ref={end} />
    </div>
  );
}
