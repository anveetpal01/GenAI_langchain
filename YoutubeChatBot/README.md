# YouTube Video Chat Assistant

Chat with the YouTube video you are watching. A Chrome side panel answers your questions **from the video's own transcript**, and every answer cites clickable timestamps like `07:55` that jump the player to the moment it was said.

```
┌─────────────── youtube.com/watch ───────────────┬──────── Side panel ─────────┐
│                                                 │ The Only 17 Minutes You…    │
│                 ▶  video player                 │ ● Ready · 8 sections        │
│                                                 │                             │
│                                                 │ You: What does he say       │
│                                                 │      about anger?           │
│                                                 │ Bot: Anger, rage and        │
│                                                 │      aggression come up     │
│                                                 │      first [07:55] …        │
│                                                 │ [Summarize] [Takeaways]     │
└─────────────────────────────────────────────────┴─────────────────────────────┘
```

## Features

- **Ask anything about the current video**: answers come only from its captions, never from the model's general knowledge.
- **Clickable timestamp citations**: every claim is cited; clicking a citation seeks the video there.
- **Streaming answers** with a Stop button.
- **Follow-up questions** ("and how do I control it?") are understood using the conversation history.
- **Quick actions**: Summarize, Key takeaways, and "What's being said right now?" (uses the current playback position).
- **Says so when the video doesn't cover it** instead of making something up.
- **Any LLM provider**: Groq, Google Gemini, OpenAI, Anthropic or local Ollama, switched by configuration. Bring your own key from the extension settings.
- **Private by default**: embeddings, search and reranking run locally; only the final prompt goes to the LLM provider. Chats and keys stay in your browser.
- **Follows YouTube navigation**: open another video and the panel switches with it, each video keeping its own chat.

## How it works

The project is a **RAG** (Retrieval-Augmented Generation) system in two parts: a Chrome extension (the face) and a local Python backend (the brain).

```
 Chrome extension                              Backend (FastAPI, localhost:8000)
 ────────────────                              ─────────────────────────────────
 content script ─ video id, time, seek
        │
 side panel ── POST /index ─────────────────►  captions → timestamped chunks → embeddings
        │                                       → Chroma vector store + BM25 keyword index
        │
        └──── POST /chat ───────────────────►  guardrails → rewrite follow-up → hybrid search
              ◄── SSE: token… citations done    → rerank → prompt → LLM (streamed) → check citations
```

**Indexing** (once per video): the transcript is fetched from YouTube's captions (manual English first, then auto-generated, then any language), merged into chunks of at most 400 tokens that keep their start and end times, embedded locally with `BAAI/bge-small-en-v1.5`, and stored in Chroma plus a BM25 index.

**Answering** (every question):

1. **Guardrails**: message length, control characters, prompt-injection patterns, 20 messages/minute rate limit.
2. **History-aware query**: a follow-up is rewritten into a standalone question.
3. **Hybrid retrieval**: dense search (meaning) and BM25 (exact words) each return their top 20, fused with Reciprocal Rank Fusion.
4. **MMR** removes near-duplicate chunks; a **cross-encoder reranker** (`BAAI/bge-reranker-base`) picks the best 5.
5. **Prompt**: the chunks are numbered `[1] (07:55-10:35) …` within a token budget, and the model is told to answer only from them and cite by number.
6. **Generation**: the answer streams token by token over Server-Sent Events.
7. **Output checks**: citations pointing to no excerpt are removed, the rest become timestamp chips; system-prompt leaks are blocked. The exchange is saved to SQLite.

Summary questions use excerpts spread across the whole video, and "right now" questions use the captions around the current playback time.

## Tech stack

| Layer | Technology |
|---|---|
| Extension | Chrome Manifest V3, Side Panel API, Preact, TypeScript, Vite |
| API | FastAPI, Uvicorn, Pydantic v2, `sse-starlette` |
| RAG orchestration | LangChain (`init_chat_model`) |
| Transcript | `youtube-transcript-api` |
| Embeddings / reranker | `sentence-transformers` with `bge-small-en-v1.5` / `bge-reranker-base` (local, CPU) |
| Search | Chroma (vectors), `rank-bm25` (keywords) |
| Memory | SQLite |
| Tokens | `tiktoken` |
| Tests | `pytest` |

## Quick start

You need Python 3.11+, Node.js 20+, Chrome 114+, and an API key for at least one LLM provider (Gemini and Groq have free tiers).

```bash
cd YoutubeChatBot
python -m venv venv
venv\Scripts\python.exe -m pip install -r backend/requirements.txt
copy backend\.env.example backend\.env
```

Set `LLM_PROVIDER`, `LLM_MODEL` and the provider's key (for example `GOOGLE_API_KEY`) in `backend/.env`, then start the backend:

```bash
cd backend
..\venv\Scripts\python.exe -m uvicorn app.main:app --port 8000
```

Wait until http://localhost:8000/readyz shows `"ready": true` (the first start downloads about 1.1 GB of models). Then build the extension:

```bash
cd ../extension
npm install
npm run build
```

In Chrome, open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and choose `extension/dist`. Open a YouTube video and click the extension icon.

Full setup, API reference and development workflow: **[setup.md](setup.md)**.

## Project structure

```
YoutubeChatBot/
├── backend/
│   ├── app/
│   │   ├── transcript.py   fetch captions, typed errors
│   │   ├── chunking.py     timestamp-aware, token-based chunks
│   │   ├── config.py       all settings (overridable in .env)
│   │   ├── models.py       local embedding + reranker models
│   │   ├── indexing.py     Chroma + BM25 index per video, job status
│   │   ├── retrieval.py    hybrid search, RRF, MMR, reranking
│   │   ├── prompts.py      versioned prompts
│   │   ├── llm.py          provider-agnostic chat model
│   │   ├── guardrails.py   input checks, rate limit, citation checks
│   │   ├── memory.py       SQLite chat history
│   │   ├── rag.py          the chat pipeline, SSE events
│   │   └── main.py         FastAPI routes
│   ├── tests/              unit tests
│   ├── requirements.txt
│   └── .env.example
├── extension/
│   ├── public/manifest.json
│   └── src/
│       ├── background/     side panel only on watch pages
│       ├── content/        runs inside YouTube: video id, seek, playback time
│       ├── sidepanel/      chat UI
│       ├── options/        settings page
│       ├── shared/         backend client, SSE reader, storage
│       └── dev/            fake chrome API for `npm run dev`
├── docs/SRS.md             requirements, design and decision log
├── setup.md                detailed setup and API reference
└── rag-using-langchain.ipynb   the original prototype
```

## API at a glance

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/videos/{id}/index` | Index a video (idempotent; `?force=true` rebuilds) |
| `GET` | `/v1/videos/{id}/status` | `not_indexed` / `indexing` / `ready` / `failed` |
| `POST` | `/v1/videos/{id}/chat` | Ask a question; answer streams as SSE |
| `GET` / `DELETE` | `/v1/videos/{id}/sessions/{session_id}` | Read / clear chat history |
| `GET` | `/healthz`, `/readyz` | Liveness and readiness |

Interactive docs at http://localhost:8000/docs while the backend runs.

## Status and roadmap

| Version | Scope | Status |
|---|---|---|
| v0.1 | Notebook pipeline turned into a tested backend package | Done (Ragas baseline still to record) |
| v0.2 | Timestamp chunks, hybrid + MMR + rerank, citations, guardrails, memory | Done |
| v0.3 | FastAPI + SSE, Chrome extension, settings page | Done; Docker packaging pending |
| v1.0 | Ragas evaluation in CI, LangSmith tracing, Chrome Web Store package | Planned |
| v2.0 | Whisper for videos without captions, agentic tools (LangGraph), multi-video chat | Planned |

Known limitations:

- Videos without captions can't be used yet.
- Retrieval and the reranker are tuned for English.
- The backend runs on your machine; there is no hosted version.
- Free-tier LLMs can be slow or temporarily overloaded; switch model or provider in the extension settings.

Requirements, design decisions and their reasons are in **[docs/SRS.md](docs/SRS.md)**.
