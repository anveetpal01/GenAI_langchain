# YouTube Video Chat Assistant: backend

FastAPI + RAG backend for the Chrome extension. Design and requirements: [`docs/SRS.md`](docs/SRS.md).

## Run

```bash
cd YoutubeChatBot
python -m venv venv
venv\Scripts\python.exe -m pip install -r backend/requirements.txt
copy backend\.env.example backend\.env   # then set LLM_PROVIDER + its API key
cd backend
..\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

The first start downloads the embedding model (~130 MB) and the reranker (~1 GB) into the Hugging Face cache. `GET /readyz` returns 200 once they are loaded.

## Try it

```bash
curl -X POST localhost:8000/v1/videos/IYFzj-7wGA0/index
curl localhost:8000/v1/videos/IYFzj-7wGA0/status
curl -N -X POST localhost:8000/v1/videos/IYFzj-7wGA0/chat -H "Content-Type: application/json" -d "{\"session_id\":\"demo\",\"message\":\"What does the speaker say about anger?\"}"
```

Interactive docs: http://localhost:8000/docs

## API

| Method | Path | Notes |
|---|---|---|
| `POST` | `/v1/videos/{id}/index?force=false` | 202. Idempotent; runs in the background |
| `GET` | `/v1/videos/{id}/status` | `not_indexed` / `indexing` / `ready` / `failed` |
| `POST` | `/v1/videos/{id}/chat` | Body `{session_id, message, playback_time?}`. SSE events: `token`, `citations`, `done`, `error` |
| `GET` / `DELETE` | `/v1/videos/{id}/sessions/{session_id}` | Chat history |
| `GET` | `/healthz`, `/readyz` | Liveness / readiness |

Optional headers on `/chat` (BYOK): `X-LLM-Provider`, `X-LLM-Model`, `X-LLM-API-Key`.
Errors are RFC 9457 `application/problem+json` with a stable `code`.

## Pipeline

```
index:  captions (manual EN > auto EN > any) -> <=400-token chunks with timestamps
        -> bge-small embeddings -> Chroma  (+ BM25 corpus in data/videos/<id>.json)
chat:   guardrails -> condense follow-up with history -> dense top-20 + BM25 top-20
        -> RRF -> MMR -> bge-reranker top-5 -> token-budgeted prompt -> streamed LLM
        -> citation check -> SQLite history
```

| Module | Responsibility |
|---|---|
| `app/transcript.py` | Fetch captions, typed `TranscriptError` |
| `app/chunking.py` | Timestamp-aware, token-based chunking |
| `app/indexing.py` | Chroma + BM25 + metadata, idempotent indexing, job status |
| `app/retrieval.py` | Hybrid search, RRF, MMR, reranking, per-stage metrics |
| `app/prompts.py` | Versioned prompts |
| `app/llm.py` | Provider-agnostic `init_chat_model` factory |
| `app/guardrails.py` | Input checks, rate limit, citation validation, prompt-leak check |
| `app/memory.py` | SQLite chat history |
| `app/rag.py` | Chat orchestration and SSE events |
| `app/main.py` | FastAPI routes, CORS, error format |

All tunables live in `app/config.py` and can be overridden in `backend/.env`.

## Tests

```bash
cd backend
..\venv\Scripts\python.exe -m pytest -q tests
```



# YouTube Video Chat: Chrome extension

Side-panel chat for the YouTube video you are watching (Manifest V3, Preact + TypeScript + Vite).
Needs the backend running: see the backend section above.

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
