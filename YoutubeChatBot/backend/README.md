# YouTube Video Chat Assistant: backend

FastAPI + RAG backend for the Chrome extension. Design and requirements: [`../docs/SRS.md`](../docs/SRS.md).

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
