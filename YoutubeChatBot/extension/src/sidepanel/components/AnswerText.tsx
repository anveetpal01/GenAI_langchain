// Renders an answer: a small, safe subset of Markdown (paragraphs, "-"/"*"
// bullets, **bold**) plus [n] citation markers as clickable timestamp chips.
// Builds Preact nodes only, never innerHTML, so model output can't inject HTML.
import type { ComponentChildren } from "preact";
import type { Citation } from "../../shared/storage";

interface Props {
  text: string;
  citations?: Citation[];
  onSeek: (seconds: number) => void;
}

const INLINE = /(\*\*[^*]+\*\*|\[\d+\])/g;

function renderInline(text: string, props: Props): ComponentChildren[] {
  const byN = new Map((props.citations ?? []).map((c) => [c.n, c]));
  return text.split(INLINE).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    }
    const marker = /^\[(\d+)\]$/.exec(part);
    if (marker) {
      const citation = byN.get(Number(marker[1]));
      // While streaming, citations aren't known yet: show the raw marker dimmed.
      if (!citation) return <span key={i} class="cite-pending">{part}</span>;
      const start = citation.timestamp.split("-")[0];
      return (
        <button
          key={i}
          type="button"
          class="cite"
          title={citation.text}
          aria-label={`Jump to ${start} in the video`}
          onClick={() => props.onSeek(citation.start_s)}
        >
          {start}
        </button>
      );
    }
    return part;
  });
}

export function AnswerText(props: Props) {
  const blocks: ComponentChildren[] = [];
  let bullets: string[] = [];

  const flushBullets = () => {
    if (bullets.length) {
      blocks.push(
        <ul key={`ul${blocks.length}`}>
          {bullets.map((b, i) => (
            <li key={i}>{renderInline(b, props)}</li>
          ))}
        </ul>,
      );
      bullets = [];
    }
  };

  for (const line of props.text.split("\n")) {
    const bullet = /^\s*[-*]\s+(.*)$/.exec(line);
    if (bullet) {
      bullets.push(bullet[1]);
      continue;
    }
    flushBullets();
    if (line.trim()) blocks.push(<p key={`p${blocks.length}`}>{renderInline(line, props)}</p>);
  }
  flushBullets();

  return <div class="answer">{blocks}</div>;
}
