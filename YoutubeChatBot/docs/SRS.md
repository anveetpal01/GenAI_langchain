# Software Requirements Specification (SRS)
## YouTube Video Chat Assistant — Chrome Extension + RAG Backend

| Field | Value |
|---|---|
| Version | 1.0 (Draft) |
| Date | 2026-10-14 |
| Author | Anveet Pal |
| Status | Decisions approved via stakeholder Q&A; ready for implementation planning |

---

## 1. Introduction

### 1.1 Purpose
This document specifies the requirements for a Chrome extension that lets a user chat with the YouTube video they are currently watching. It covers the browser extension, the local Python RAG (Retrieval-Augmented Generation) backend, evaluation, and operational requirements. It is the single source of truth for scope, and every requirement is traceable to `improvements.txt` (Section 10).

### 1.2 Scope
**In scope (v1):** Chrome extension (Manifest V3) with side-panel chat on youtube.com, local FastAPI backend, caption-based ingestion, hybrid retrieval with reranking, grounded answers with clickable timestamp citations, per-video conversation memory, Ragas + LangSmith evaluation, Docker packaging, BYOK (Bring Your Own Key) for LLM providers.

**Out of scope (v1, planned for v2):** Whisper / visual multimodal ingestion, agentic tool use (LangGraph), query rewriting and multi-query, contextual compression, domain routing, hosted multi-tenant backend, Firefox/Edge store listing.

### 1.3 Definitions
| Term | Meaning |
|---|---|
| RAG | Retrieval-Augmented Generation |
| Chunk | A transcript segment (<= 400 tokens) carrying start/end timestamps |
| Hybrid retrieval | BM25 (lexical) + dense vector search fused with Reciprocal Rank Fusion (RRF) |
| MMR | Maximal Marginal Relevance: diversity-aware re-selection of candidates |
| Reranker | Cross-encoder model that rescores (query, chunk) pairs |
| BYOK | User supplies their own LLM API key; nothing is shared with anyone |
| Grounded answer | Every claim is supported by retrieved transcript chunks and cited |

### 1.4 References
- `YoutubeChatBot/Problem_stmt.txt` — original problem statement
- `YoutubeChatBot/improvements.txt` — improvement backlog (source of traceability IDs)
- `YoutubeChatBot/rag-using-langchain.ipynb` — current prototype (baseline)

---

## 2. Overall Description

### 2.1 Product Perspective
```
+------------------------- Chrome (youtube.com) --------------------------+
|  Content Script --(video_id, seek)--> Side Panel UI <--> Options page    |
|       ^                                    |  SSE / JSON                 |
|  reads URL, controls <video>               v                             |
+--------------------------------------------|-----------------------------+
                                             | http://localhost:8000
+------------------------- Backend (Docker) -|-----------------------------+
| FastAPI --> Ingestion --> Chunking --> Embeddings --> Chroma (persistent) |
|   |                                                    + BM25 index      |
|   +--> Chat: Hybrid(RRF) --> MMR --> Reranker --> Prompt --> LLM -->     |
|          ^ history-aware query               citations + guardrails      |
|   Memory store (SQLite)          LangSmith tracing (opt-in)              |
+--------------------------------------------------------------------------+
```

### 2.2 Users
| User | Need |
|---|---|
| Viewer (primary) | Ask questions about the playing video, jump to the cited moment |
| Developer / Maintainer | Run, test, evaluate and extend the pipeline |

### 2.3 Operating Environment
- Chrome >= 114 (Side Panel API), Manifest V3
- Backend: Python 3.11+, Docker 24+, runs on the user's machine (CPU-only is sufficient)
- LLM: external provider via HTTPS (free tier by default)

### 2.4 Design Constraints
- **C-1** Free-tier LLM APIs by default; provider switchable via configuration only (no code change).
- **C-2** Embeddings and reranker run locally (no API cost; no data leaves the machine except the final LLM prompt).
- **C-3** User API keys never leave the user's machine except to the chosen provider.
- **C-4** Transcript source is YouTube captions only (manual or auto-generated); no audio download in v1.

### 2.5 Assumptions
- **A-1** Target videos have English captions (manual or auto). Other caption languages are indexed if present but answer quality is not guaranteed.
- **A-2** Free-tier quotas (requests/min, tokens/day) are acceptable for single-user usage.

---

## 3. Functional Requirements

Priority: **M** = Must (v1), **S** = Should (v1 if time permits), **C** = Could (v2).

### 3.1 Chrome Extension (UI) — improvements §1
| ID | Requirement | Pri |
|---|---|---|
| FR-UI-01 | Extension registers a side panel available only on `youtube.com/watch*` pages. | M |
| FR-UI-02 | Content script detects the current `video_id` from the URL, including YouTube SPA navigation (`yt-navigate-finish` event), and notifies the panel. | M |
| FR-UI-03 | On video change the panel shows index status (Not indexed -> Indexing with progress -> Ready) and triggers indexing (user setting: auto / on first question). | M |
| FR-UI-04 | Chat UI streams the answer token-by-token (SSE) with a visible Stop button. | M |
| FR-UI-05 | Citations render as clickable `[mm:ss]` chips; clicking seeks the page `<video>` element to that time. | M |
| FR-UI-06 | Per-video conversation history is shown and persisted locally (`chrome.storage.local`); a Clear chat action exists. | M |
| FR-UI-07 | Options page: backend URL, optional LLM provider/key override, auto-index toggle, backend health check with clear error states (backend down, key invalid, no captions). | M |
| FR-UI-08 | Quick actions: Summarize video, Key takeaways, "What is being said right now?" (uses current playback time). | S |
| FR-UI-09 | Accessible (keyboard navigation, ARIA roles, contrast >= 4.5:1) and follows system dark/light theme. | S |

### 3.2 Backend API
| ID | Requirement | Pri |
|---|---|---|
| FR-API-01 | `POST /v1/videos/{video_id}/index` — idempotent; returns 202 with job status; skips re-indexing if an index exists and the transcript hash is unchanged. | M |
| FR-API-02 | `GET /v1/videos/{video_id}/status` returns `{state: not_indexed / indexing / ready / failed, chunks, language, error}`. | M |
| FR-API-03 | `POST /v1/videos/{video_id}/chat` — body `{session_id, message, playback_time?}`; responds as an SSE stream of `token`, `citations`, `done`, `error` events. | M |
| FR-API-04 | `GET` and `DELETE /v1/videos/{video_id}/sessions/{session_id}` — read / clear history. | M |
| FR-API-05 | `GET /healthz` (liveness) and `GET /readyz` (models loaded, vector store reachable, LLM key present). | M |
| FR-API-06 | CORS restricted to the extension origin (`chrome-extension://<id>`) and localhost. | M |
| FR-API-07 | All requests/responses validated with Pydantic schemas; errors follow RFC 9457 Problem Details. | M |
| FR-API-08 | Optional per-request headers `X-LLM-Provider` / `X-LLM-API-Key` override `.env` defaults (BYOK from the extension). | S |

### 3.3 Indexing — improvements §3
| ID | Requirement | Pri |
|---|---|---|
| FR-IDX-01 | **Ingestion:** fetch captions via `youtube-transcript-api`; prefer manual EN, then auto-generated EN, then any language. Store `language` and `is_generated` metadata. | M |
| FR-IDX-02 | Captions unavailable / disabled / private video -> typed error; UI shows a clear, non-technical message. | M |
| FR-IDX-03 | **Text splitting:** timestamp-aware chunking. Merge caption segments into windows of at most 400 tokens (cl100k) with ~15 % overlap (60 tokens), kept below the embedding model's 512-token input limit (see D-9); each chunk carries `start_s`, `end_s`, `chunk_index`, `video_id`, `title`. No timestamp information is lost (the baseline notebook discards it by joining text). | M |
| FR-IDX-04 | Token counting uses the active LLM's tokenizer (tiktoken fallback). | M |
| FR-IDX-05 | **Vector store:** Chroma persistent client, one collection per `video_id`; embeddings via local `BAAI/bge-small-en-v1.5` (configurable). | M |
| FR-IDX-06 | A BM25 index is built per video and persisted alongside the vector collection. | M |
| FR-IDX-07 | Index metadata (transcript hash, embedding model, chunking params, created_at) is stored so stale indexes are detected and rebuilt when parameters change. | M |
| FR-IDX-08 | Indexing a 60-minute video completes in <= 30 s on a 4-core CPU once models are warm. | S |

### 3.4 Retrieval — improvements §4
| ID | Requirement | Pri |
|---|---|---|
| FR-RET-01 | **Hybrid retrieval:** dense (Chroma, top-20) and BM25 (top-20) fused with RRF (k = 60). | M |
| FR-RET-02 | **MMR** on the fused candidate set (lambda = 0.7) removes near-duplicate chunks. | M |
| FR-RET-03 | **Reranking:** local cross-encoder `BAAI/bge-reranker-base` rescores top-20 and selects top-5 (configurable). | M |
| FR-RET-04 | **History-aware query:** follow-up questions are condensed with chat history into a standalone query before retrieval (one LLM call; skipped when there is no history). | M |
| FR-RET-05 | Retrieval emits structured metrics: latency per stage, candidate counts, reranker scores. | M |
| FR-RET-06 | Pre-retrieval **query rewriting** and **multi-query** generation (3 variants, union, RRF) behind a feature flag. | C |
| FR-RET-07 | **Domain-aware routing:** classify intent (summary / specific fact / timestamp-local / off-topic / chit-chat) and pick a strategy (summary -> map-reduce over all chunks; off-topic -> polite refusal without retrieval). | C |
| FR-RET-08 | **Contextual compression:** keep only relevant sentences from each selected chunk before prompting. | C |

### 3.5 Augmentation — improvements §5
| ID | Requirement | Pri |
|---|---|---|
| FR-AUG-01 | **Prompt templating:** versioned prompt files (YAML/Jinja) under `app/prompts/`: system, QA, condense-question, summary. Prompt version is logged with every trace. | M |
| FR-AUG-02 | **Answer grounding:** instruct the model to answer only from context; each chunk is injected as `[n] (mm:ss-mm:ss) text` so the model cites by index. | M |
| FR-AUG-03 | **Context window optimization:** token budget = model context - (system + history + answer reserve). Chunks are added in rerank order until the budget is reached; history truncated oldest-first. | M |
| FR-AUG-04 | Playback time (if provided) is injected as "The user is currently at mm:ss" to bias "right now" questions. | S |

### 3.6 Generation — improvements §6
| ID | Requirement | Pri |
|---|---|---|
| FR-GEN-01 | LLM via LangChain `init_chat_model`; provider/model from config. Default free tier: Groq (`llama-3.3-70b-versatile`). Alternates: Google Gemini (`gemini-3.5-flash`; `gemini-2.5-flash` is closed to new users), Hugging Face Inference, Ollama (local). Paid providers (OpenAI, Anthropic) enabled by config only. | M |
| FR-GEN-02 | Streaming output; first token <= 3 s (p50) after retrieval. | M |
| FR-GEN-03 | **Citations:** model emits `[n]` markers; backend maps them to `{start_s, end_s, text}` and streams a `citations` event. Unmapped markers are stripped. | M |
| FR-GEN-04 | When retrieval confidence is low (top reranker score below threshold) or the model says the answer is not in the video, the response says so explicitly and offers a summary instead. | M |
| FR-GEN-05 | **Input guardrails:** max message length, prompt-injection pattern screening, control-character stripping, per-session rate limit (e.g. 20 req/min). | M |
| FR-GEN-06 | **Output guardrails:** citation validation (every `[n]` exists), no leaked system prompt, optional LLM-based faithfulness check behind a flag. | M |
| FR-GEN-07 | Provider failure / quota exhausted -> automatic fallback to the next configured provider, surfaced to the UI as a notice. | S |

### 3.7 Memory & System Design — improvements §7
| ID | Requirement | Pri |
|---|---|---|
| FR-MEM-01 | Conversation memory per `(video_id, session_id)` persisted in SQLite via LangChain message history; last N turns feed condensing (FR-RET-04) and prompting. | M |
| FR-MEM-02 | **Agentic (v2):** LangGraph agent with tools `search_transcript`, `summarize_section(start, end)`, `seek(timestamp)`, `web_search` (external facts clearly labelled as non-video sources). | C |
| FR-MEM-03 | **Multimodal (v2):** Whisper fallback for caption-less videos; key-frame captioning for visual questions. | C |
| FR-MEM-04 | **Multi-video (v2):** chat across a playlist/channel with per-video citations. | C |

### 3.8 Evaluation — improvements §2
| ID | Requirement | Pri |
|---|---|---|
| FR-EVAL-01 | **Ragas** reference-free evaluation (no hand-written golden dataset, see D-16): `faithfulness` and `answer_relevancy` scored by a judge LLM on a sample of real requests, across whatever videos users open. Judge LLM = configured free provider. | M |
| FR-EVAL-02 | Results written to `evaluation/results/<git-sha>.json` plus a Markdown report; CI fails if any metric drops more than 5 points vs. baseline. | M |
| FR-EVAL-03 | **LangSmith** tracing for every request when `LANGCHAIN_TRACING_V2=true`; traces tagged with `video_id`, `prompt_version`, `retrieval_strategy`. Off by default for privacy. | M |
| FR-EVAL-04 | ~~Baseline on a golden dataset~~ Dropped (D-16). Comparisons (e.g. cache/routing on vs off) use the same sampled real questions replayed through both configurations. | - |
| FR-EVAL-05 | Config switch to compare retrieval strategies (similarity-only vs hybrid + rerank) within one eval run. | S |

---

## 4. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-01 | Performance | End-to-end answer p95 <= 10 s on free-tier LLM; retrieval + rerank <= 1.5 s on CPU. |
| NFR-02 | Reliability | Indexing is idempotent and resumable; backend restarts never lose indexes or history. |
| NFR-03 | Security | API keys only in `.env` or `chrome.storage.local`, never logged. CORS locked to extension origin. `pip-audit` in CI. Extension requests minimum permissions: `sidePanel`, `storage`, `activeTab`, host `*://*.youtube.com/*`. |
| NFR-04 | Privacy | No telemetry by default. Only the prompt (query + selected chunks) reaches the LLM provider. LangSmith is opt-in. |
| NFR-05 | Observability | Structured JSON logs (request_id, video_id, stage latencies, token usage, cost estimate). `/metrics` Prometheus endpoint (Could). |
| NFR-06 | Maintainability | Layered architecture (`api` / `rag` / `infra`); >= 80 % unit-test coverage on the `rag` package; `ruff`, `mypy --strict`, `pre-commit`. |
| NFR-07 | Configurability | All tunables (chunk size, top-k, lambda, thresholds, model names) via `pydantic-settings`; `.env.example` documents each. |
| NFR-08 | Portability | `docker compose up` works on Windows/macOS/Linux; model weights cached in a named volume. |
| NFR-09 | Usability | A new user gets a first successful answer within 2 minutes of install, following the README only. |
| NFR-10 | Compatibility | Extension passes Chrome Web Store MV3 review (no remote code, CSP compliant). |

---

## 5. Technology Stack

| Layer | Choice | Rationale |
|---|---|---|
| Extension | TypeScript, Vite, Manifest V3, Side Panel API, Preact (or vanilla) | Small bundle, store-compliant, no heavy framework |
| API | FastAPI, Uvicorn, Pydantic v2, `sse-starlette` | Async, typed, streaming-native |
| Orchestration | LangChain (LCEL); LangGraph in v2 | Matches existing learning path; components are swappable |
| Embeddings | `BAAI/bge-small-en-v1.5` via `sentence-transformers` (local) | Free, fast on CPU, strong MTEB scores |
| Vector store | Chroma (persistent, local) | Zero infra, metadata filters, per-video collections |
| Lexical | `rank_bm25` via LangChain `BM25Retriever` | Hybrid recall on names and rare terms |
| Reranker | `BAAI/bge-reranker-base` (CrossEncoder, local) | Free precision boost |
| LLM | `init_chat_model` -> Groq / Gemini / HF / Ollama; OpenAI / Anthropic optional | Free by default, config-switchable (C-1) |
| Memory | SQLite + `SQLChatMessageHistory` | Durable, no extra service |
| Evaluation | Ragas, LangSmith | Industry-standard RAG metrics and tracing |
| Tooling | `uv`, `ruff`, `mypy`, `pytest`, `pre-commit`, GitHub Actions, Docker multi-stage | Reproducible, CI-enforced quality |

---

## 6. Data Model

```
Video    { video_id PK, title, language, is_generated, transcript_hash,
           chunk_count, embedding_model, chunk_params_json, created_at, updated_at }
Chunk    { id, video_id, chunk_index, start_s, end_s, text, token_count }   # Chroma document
Session  { session_id PK, video_id FK, created_at, last_active_at }
Message  { id PK, session_id FK, role, content, citations_json, prompt_version,
           latency_ms, tokens_in, tokens_out, created_at }
Request  { id PK, created_at, video_id, session_id, status, error_code, route, provider, model,
           prompt_version, cache_hit, tokens_in, tokens_out, cost_usd, usage_estimated,
           latency_ms, ttft_ms, retrieval_ms, llm_ms, calls_json }   # per-request cost/latency log (D-13)
EvalRun  { git_sha, dataset_version, strategy, faithfulness, answer_relevancy,
           context_precision, context_recall, created_at }
```

---

## 7. Repository Layout

```
YoutubeChatBot/
+-- backend/
|   +-- app/
|   |   +-- api/            # routers, schemas, SSE
|   |   +-- core/           # settings, logging, errors
|   |   +-- rag/
|   |   |   +-- ingest/     # transcript fetch, timestamp-aware chunking
|   |   |   +-- retrieve/   # hybrid, mmr, rerank, condense-question
|   |   |   +-- generate/   # prompts, llm factory, citations, guardrails
|   |   |   +-- memory/
|   |   +-- infra/          # chroma, sqlite, bm25 persistence
|   +-- tests/
|   +-- pyproject.toml
|   +-- Dockerfile
+-- extension/
|   +-- src/{background,content,sidepanel,options,shared}
|   +-- manifest.json
|   +-- package.json
+-- evaluation/
|   +-- samples/        # sampled real questions for replay
|   +-- run_ragas.py
|   +-- results/
+-- docs/
|   +-- SRS.md
|   +-- adr/                # Architecture Decision Records
+-- docker-compose.yml
+-- .env.example
+-- README.md
```

---

## 8. Release Plan

| Version | Contents | Exit criteria |
|---|---|---|
| **v0.1 Baseline** | Notebook pipeline refactored into `backend/app/rag` with tests; Ragas baseline recorded | Baseline metrics committed |
| **v0.2 Core RAG** | Timestamp chunking, Chroma + BM25, hybrid + MMR + rerank, citations, guardrails, memory | `context_precision` >= baseline + 10 pts; all Must items in FR-IDX / RET / AUG / GEN pass |
| **v0.3 API + Extension** | FastAPI with SSE, side panel, seek-to-citation, options page, Docker | NFR-09 met by two external testers |
| **v1.0 Release** | LangSmith tracing, CI with eval gate, README, Chrome Web Store package | All Must requirements verified; eval gate green |
| **v2.0** | FR-RET-06/07/08, FR-MEM-02/03/04, FR-GEN-07, optional hosted mode | Separate SRS revision |

---

## 9. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| Free-tier rate limits / model deprecation | Failed answers | Provider fallback chain (FR-GEN-07); config-only switch |
| `youtube-transcript-api` breaks on YouTube changes | No ingestion | Pin version; integration test in CI; typed error with clear UI message |
| Local models slow on low-end CPUs | Poor UX | Small models by default; lazy load; warm-up on startup; reranker can be disabled |
| Prompt injection via transcript content | Unsafe output | Transcript treated as data; system-prompt isolation; output guardrails |
| Chrome Web Store review rejection | Distribution blocked | MV3 compliance, minimal permissions, privacy policy |

---

## 10. Traceability Matrix (`improvements.txt` -> SRS)

| improvements.txt | Requirement IDs | Version |
|---|---|---|
| 1 UI / plugin | FR-UI-01..09, FR-API-01..08 | v0.3 |
| 2a Ragas | FR-EVAL-01, 02, 04, 05 | v0.1 -> v1.0 |
| 2b LangSmith | FR-EVAL-03 | v1.0 |
| 3a Document ingestion | FR-IDX-01, 02 | v0.2 |
| 3b Text splitting | FR-IDX-03, 04 | v0.2 |
| 3c Vector store | FR-IDX-05, 06, 07 | v0.2 |
| 4a-i Query rewriting | FR-RET-06 | v2.0 |
| 4a-ii Multi-query | FR-RET-06 | v2.0 |
| 4a-iii Domain-aware routing | FR-RET-07 | v2.0 |
| 4b-i MMR | FR-RET-02 | v0.2 |
| 4b-ii Hybrid retrieval | FR-RET-01 | v0.2 |
| 4b-iii Reranking | FR-RET-03 | v0.2 |
| 4c-i Contextual compression | FR-RET-08 | v2.0 |
| 5a Prompt templating | FR-AUG-01 | v0.2 |
| 5b Answer grounding | FR-AUG-02, FR-GEN-04 | v0.2 |
| 5c Context window optimization | FR-AUG-03 | v0.2 |
| 6a Answer with citation | FR-GEN-03, FR-UI-05 | v0.2 / v0.3 |
| 6b Guardrailing | FR-GEN-05, 06 | v0.2 |
| 7a Multimodal | FR-MEM-03 | v2.0 |
| 7b Agentic | FR-MEM-02 | v2.0 |
| 7c Memory based | FR-MEM-01, FR-RET-04 | v0.2 |

---

## 11. Decision Log (stakeholder Q&A, 2026-10-14)

| # | Decision |
|---|---|
| D-1 | Chrome Extension (MV3) + Python FastAPI backend |
| D-2 | Provider-agnostic LLM layer; free-tier providers by default, paid enabled later via config only |
| D-3 | Chroma persistent, local, one collection per video |
| D-4 | Captions only in v1 with graceful error; Whisper deferred to v2 |
| D-5 | v1 retrieval = Hybrid (BM25 + dense, RRF) + MMR + cross-encoder reranking |
| D-6 | Evaluation = Ragas (offline metrics) + LangSmith (tracing) |
| D-7 | Conversation memory in v1; agentic behaviour in v2 |
| D-8 | Local Docker deployment with BYOK; no hosted backend in v1 |
| D-9 | (2026-10-25) Chunk size reduced from ~600-800 to <= 400 tokens with 60-token overlap: `bge-small-en-v1.5` embeds at most 512 tokens and silently truncates the rest. Revisit if the embedding model changes. |
| D-10 | (2026-10-25) Extension UI built with Preact + TypeScript + Vite (Section 5 option chosen over vanilla). |
| D-11 | (2026-10-25) FR-UI-03 default: index on the first question, not on video change; auto-index on video change stays available as an option. |
| D-12 | (2026-10-26) Prompt `qa-v2`: video details (title, channel from YouTube oEmbed, length) are a second allowed answer source besides transcript excerpts, cited by name instead of [n]; questions about them skip the not-in-video shortcut. |
| D-13 | (2026-10-27) Semantic caching (routing later dropped, D-18), designed to run both locally and live. Storage behind one interface selected by `DATABASE_URL`: SQLite locally, Postgres + pgvector live (cache vectors, chat history, request/cost log). Cache is shared across users per video (videos are public). |
| D-14 | ~~(2026-10-27) Router = small-LLM classifier.~~ Superseded by D-18: routing removed for now. |
| D-15 | (2026-10-27) Dashboard, when built, is a built-in FastAPI `/dashboard` page. Postponed (D-18); the `requests` table already holds the numbers. |
| D-16 | (2026-10-28) No golden dataset: the bot serves any video, so evaluation must not depend on hand-written answers for a few videos. Quality is tracked with reference-free metrics on sampled real requests, user thumbs up/down, and replaying sampled questions with features on vs off. |
| D-17 | (2026-10-28) Semantic cache hit = cosine >= 0.85 AND content-word Jaccard >= 0.5 (or identical normalised text). Cosine alone was unsafe: 'Summarize the first half' scored 0.900 vs 'Summarize this video', while a true paraphrase scored 0.856. Lookups are always per video + scope (transcript hash, prompt version, embedding model), so similarity is computed in numpy over a few hundred rows; pgvector is not needed for the cache. Only first-turn answers are stored; time-local questions are never cached; `X-Cache: bypass` skips the cache for on/off replays (D-16). |
| D-18 | (2026-10-29) Scope kept simple for v1 (first project): cost-aware routing removed (it was half-built and its live test found a bug); dashboard, Ragas judge, thumbs up/down and replay script postponed. Kept: extension, RAG pipeline, semantic cache, per-request cost/latency log, SQLite/Postgres layer. Video-details questions are recognised by intent patterns across video types (who is the singer/guest/teacher, who made this, title, channel), not by the bare word 'speaker'. |
