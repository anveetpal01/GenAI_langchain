"""FastAPI app (SRS §3.2). Run from the backend/ folder:

    uvicorn app.main:app --port 8000
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import BackgroundTasks, FastAPI, Header, Path, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from app import indexing, memory, models
from app.config import get_settings
from app.guardrails import GuardrailError
from app.llm import LLMConfigError, has_key, resolve
from app.rag import ChatError, prepare_chat, stream_chat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

VideoId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{11}$", description="11-character YouTube video ID")]
SessionId = Annotated[str, Path(pattern=r"^[A-Za-z0-9_-]{1,64}$")]

_models_ready = threading.Event()


def _warm_up() -> None:
    try:
        models.warm_up()
        _models_ready.set()
        logger.info("Models loaded")
    except Exception:
        logger.exception("Model warm-up failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models in the background so /healthz answers immediately.
    threading.Thread(target=_warm_up, daemon=True).start()
    yield


app = FastAPI(title="YouTube Video Chat Assistant", version="0.3.0", lifespan=lifespan)

_settings = get_settings()
if _settings.allowed_extension_ids:
    _ext_origins = "|".join(_settings.allowed_extension_ids)
else:
    _ext_origins = "[a-p]{32}"  # any extension (development)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=rf"^(chrome-extension://({_ext_origins})|http://(localhost|127\.0\.0\.1)(:\d+)?)$",
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-LLM-Provider", "X-LLM-Model", "X-LLM-API-Key"],
)


# ---------------------------------------------------------------- errors (RFC 9457)

def problem(status_code: int, code: str, detail: str, **extra) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        media_type="application/problem+json",
        content={
            "type": f"urn:yt-chatbot:error:{code}",
            "title": code.replace("_", " ").capitalize(),
            "status": status_code,
            "detail": detail,
            "code": code,
            **extra,
        },
    )


@app.exception_handler(GuardrailError)
@app.exception_handler(ChatError)
async def _app_error(_: Request, exc: GuardrailError | ChatError):
    return problem(exc.status, exc.code, exc.message)


@app.exception_handler(LLMConfigError)
async def _llm_config_error(_: Request, exc: LLMConfigError):
    return problem(status.HTTP_400_BAD_REQUEST, exc.code, exc.message)


@app.exception_handler(RequestValidationError)
async def _validation_error(_: Request, exc: RequestValidationError):
    errors = [{"loc": e["loc"], "msg": e["msg"]} for e in exc.errors()]
    return problem(status.HTTP_422_UNPROCESSABLE_ENTITY, "validation_error", "Invalid request.", errors=errors)


# ---------------------------------------------------------------- schemas

class VideoStatus(BaseModel):
    video_id: str
    state: str  # not_indexed | indexing | ready | failed
    chunks: int | None = None
    language: str | None = None
    title: str | None = None
    error: dict | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    message: str = Field(min_length=1, max_length=10_000)  # real limit enforced by guardrails
    playback_time: float | None = Field(default=None, ge=0)


class HistoryMessage(BaseModel):
    role: str
    content: str
    citations: list[dict]
    created_at: str


class SessionHistory(BaseModel):
    video_id: str
    session_id: str
    messages: list[HistoryMessage]


# ---------------------------------------------------------------- routes

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    s = get_settings()
    provider, model = resolve(None, None)
    checks = {
        "models_loaded": _models_ready.is_set(),
        "vector_store": indexing.chroma_heartbeat(),
        "llm_key_present": has_key(provider),
    }
    body = {"ready": all(checks.values()), "checks": checks, "llm": {"provider": provider, "model": model},
            "rerank_enabled": s.rerank_enabled}
    return JSONResponse(body, status_code=200 if body["ready"] else 503)


@app.post("/v1/videos/{video_id}/index", status_code=status.HTTP_202_ACCEPTED, response_model=VideoStatus)
def index_video(video_id: VideoId, background: BackgroundTasks, force: bool = False):
    current = indexing.get_status(video_id)
    if current["state"] == "ready" and not force:
        return VideoStatus(video_id=video_id, **current)
    if indexing.try_start_job(video_id):
        background.add_task(indexing.run_index_job, video_id, force)
    return VideoStatus(video_id=video_id, **indexing.get_status(video_id))


@app.get("/v1/videos/{video_id}/status", response_model=VideoStatus)
def video_status(video_id: VideoId):
    return VideoStatus(video_id=video_id, **indexing.get_status(video_id))


@app.post("/v1/videos/{video_id}/chat")
async def chat(
    video_id: VideoId,
    body: ChatRequest,
    x_llm_provider: Annotated[str | None, Header()] = None,
    x_llm_model: Annotated[str | None, Header()] = None,
    x_llm_api_key: Annotated[str | None, Header()] = None,
):
    ctx = prepare_chat(
        video_id, body.session_id, body.message, body.playback_time,
        provider=x_llm_provider, model=x_llm_model, api_key=x_llm_api_key,
    )

    async def events():
        async for ev in stream_chat(ctx):
            yield {"event": ev.event, "data": json.dumps(ev.data, ensure_ascii=False)}

    return EventSourceResponse(events())


@app.get("/v1/videos/{video_id}/sessions/{session_id}", response_model=SessionHistory)
def get_session(video_id: VideoId, session_id: SessionId):
    return SessionHistory(
        video_id=video_id, session_id=session_id, messages=memory.get_history(video_id, session_id)
    )


@app.delete("/v1/videos/{video_id}/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(video_id: VideoId, session_id: SessionId):
    memory.clear_history(video_id, session_id)
