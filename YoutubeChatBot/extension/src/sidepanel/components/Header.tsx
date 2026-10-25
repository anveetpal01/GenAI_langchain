import type { VideoStatus } from "../../shared/api";

interface Props {
  title: string | null;
  status: VideoStatus | null;
  busy: boolean;
  onReindex: () => void;
  onClear: () => void;
  onSettings: () => void;
}

function statusLabel(status: VideoStatus | null): { text: string; tone: string } {
  switch (status?.state) {
    case "ready":
      return { text: `Ready · ${status.chunks} sections`, tone: "ok" };
    case "indexing":
      return { text: "Reading the video…", tone: "warn" };
    case "failed":
      return { text: status.error?.message ?? "Indexing failed", tone: "danger" };
    case "not_indexed":
      return { text: "Will read the video on your first question", tone: "muted" };
    default:
      return { text: "Checking…", tone: "muted" };
  }
}

export function Header({ title, status, busy, onReindex, onClear, onSettings }: Props) {
  const label = statusLabel(status);
  return (
    <header class="header">
      <div class="header-text">
        <h1 class="title" title={title ?? undefined}>
          {status?.title ?? title ?? "YouTube video"}
        </h1>
        <p class={`status status-${label.tone}`} role="status">
          <span class="dot" aria-hidden="true" />
          {label.text}
        </p>
      </div>
      <div class="header-actions">
        <button type="button" class="icon-btn" onClick={onReindex} disabled={busy} title="Read the video again">
          <span aria-hidden="true">↻</span>
          <span class="sr-only">Re-index video</span>
        </button>
        <button type="button" class="icon-btn" onClick={onClear} disabled={busy} title="Clear chat">
          <span aria-hidden="true">🗑</span>
          <span class="sr-only">Clear chat</span>
        </button>
        <button type="button" class="icon-btn" onClick={onSettings} title="Settings">
          <span aria-hidden="true">⚙</span>
          <span class="sr-only">Settings</span>
        </button>
      </div>
    </header>
  );
}
