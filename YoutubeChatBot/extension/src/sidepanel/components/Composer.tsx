import { useState } from "preact/hooks";

export const QUICK_ACTIONS = [
  { label: "Summarize", message: "Summarize this video" },
  { label: "Key takeaways", message: "What are the key takeaways of this video?" },
  { label: "What's being said now?", message: "What is being said right now in the video?" },
];

interface Props {
  disabled: boolean;
  streaming: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}

export function Composer({ disabled, streaming, onSend, onStop }: Props) {
  const [text, setText] = useState("");

  const submit = () => {
    const message = text.trim();
    if (!message || disabled || streaming) return;
    onSend(message);
    setText("");
  };

  return (
    <div class="composer">
      <div class="quick-actions">
        {QUICK_ACTIONS.map((a) => (
          <button
            key={a.label}
            type="button"
            class="btn chip-btn"
            disabled={disabled || streaming}
            onClick={() => onSend(a.message)}
          >
            {a.label}
          </button>
        ))}
      </div>
      <form
        class="input-row"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        <label for="question" class="sr-only">
          Ask about this video
        </label>
        <textarea
          id="question"
          rows={2}
          placeholder="Ask about this video…"
          value={text}
          disabled={disabled}
          onInput={(e) => setText(e.currentTarget.value)}
          onKeyDown={(e) => {
            // Enter sends, Shift+Enter adds a new line.
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
        />
        {streaming ? (
          <button type="button" class="btn send-btn" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" class="btn btn-primary send-btn" disabled={disabled || !text.trim()}>
            Send
          </button>
        )}
      </form>
    </div>
  );
}
