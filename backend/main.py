"""FastAPI backend — docs/ARCHITECTURE.md §2, §4. Thin API layer: owns session lifecycle and request
validation (Pydantic), invokes the LangGraph agent; no business logic beyond that, per the doc.
"""

import time
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from agent.graph import run_agent
from backend.schemas import (
    HistoryResponse,
    HistoryTurn,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from backend.session import SessionNotFoundError, session_store
from ingestion import IngestionError, corpus_store, load_csv, load_pdf, load_txt
from models.faiss_index import warm_up_embedding_model
from models.registry import get_sentiment_pipeline, get_summarization_pipeline, get_zero_shot_pipeline
from observability.logging import get_logger
from observability.metrics import metrics_registry
from tools.profile_dataset import profile_dataset

_startup_logger = get_logger("startup")


def _warm_up_all_models() -> None:
    """Eagerly load every default model once at process startup, per
    docs/LATENCY_AND_PERFORMANCE.md §4's originally-intended "loaded once at process startup" option --
    only the lazy "on first use" half was ever built until now. This is the actual fix for the concurrent-
    first-load race documented in LOAD_TEST_RESULTS.md: locking each cache individually (models/registry.py,
    models/faiss_index.py) was not enough, because the race was in transformers' own shared internal
    lazy-import state, not in either cache. Loading everything before the server accepts any traffic means
    no two requests can ever both be first to load the same model, because nothing is ever first at
    request time -- eliminating the race entirely rather than narrowing it.
    """
    start = time.perf_counter()
    get_sentiment_pipeline()
    get_zero_shot_pipeline()
    get_summarization_pipeline()
    warm_up_embedding_model()
    duration_ms = (time.perf_counter() - start) * 1000
    _startup_logger.info("model_warmup", duration_ms=round(duration_ms, 2))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _warm_up_all_models()
    yield


app = FastAPI(title="TextInsight API", lifespan=lifespan)

# Rate limiting (item 4, 2026-09-02 scope revision): protects this API from being hammered by one client,
# separate from llm/client.py's Groq-side retry/backoff handling (docs/SECURITY_AND_RELIABILITY.md §6-7),
# which protects against the *provider's* limits, not ours. In-memory (per-process) limiter — this
# single-instance deployment's scope doesn't need a shared Redis-backed limiter across replicas.
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

_request_logger = get_logger("request")


class RequestLoggingMiddleware:
    """Every request (item 3, 2026-09-02 scope revision): logs endpoint/method/status/duration and feeds
    the /metrics counters. session_id is bound as a contextvar by the handlers below (upload/query/history)
    as soon as it's known, so it's merged into this log line automatically.

    Deliberately a raw ASGI middleware, not `@app.middleware("http")` (Starlette's `BaseHTTPMiddleware`):
    `BaseHTTPMiddleware.call_next` runs the downstream route handler in a separate anyio task, so a
    contextvar bound inside the handler (session_id) does not propagate back to code running after
    `call_next` returns — a well-known Starlette gotcha that silently dropped session_id from this exact
    log line during testing. Awaiting `self.app(...)` directly, in the same task, avoids it.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(endpoint=scope["path"], method=scope["method"])

        status_holder: dict[str, int] = {}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
            await send(message)

        start = time.perf_counter()
        await self.app(scope, receive, send_wrapper)
        duration_ms = (time.perf_counter() - start) * 1000

        status = status_holder.get("status", 500)
        metrics_registry.record_request(duration_ms, is_error=status >= 400)
        _request_logger.info("request", status=status, duration_ms=round(duration_ms, 2))


app.add_middleware(RequestLoggingMiddleware)


# Single-user local/demo deployment (docs/PROJECT_SPEC.md §10: no multi-tenant/enterprise auth in scope) —
# the frontend runs as a separate process/container per docs/TECH_STACK.md, so permissive CORS here is a
# deliberate, scoped choice, not an oversight.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_LOADERS = {"csv": load_csv, "txt": load_txt, "pdf": load_pdf}


def _select_loader(filename: str):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    loader = _LOADERS.get(ext)
    if loader is None:
        raise IngestionError(f"Unsupported file type '.{ext}' — expected .csv, .txt, or .pdf.")
    return loader


def _serialize_tool_results(tool_results: dict) -> dict:
    return {
        name: (result.model_dump() if hasattr(result, "model_dump") else result)
        for name, result in tool_results.items()
    }


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...), session_id: str | None = Form(None)):
    content = await file.read()

    try:
        loader = _select_loader(file.filename or "")
        corpus = loader(content, file.filename)
    except IngestionError as exc:
        # Ingestion errors -> structured 4xx, agent never invoked — docs/ARCHITECTURE.md §11.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    corpus_store.put(corpus)

    if session_id is None or not session_store.exists(session_id):
        session_id = session_store.create()
    structlog.contextvars.bind_contextvars(session_id=session_id)

    profile_output = profile_dataset(corpus_ref=corpus.corpus_ref)
    profile_dict = profile_output.model_dump()
    session_store.set_corpus(session_id, corpus.corpus_ref, corpus.source_filename, profile_dict)

    return UploadResponse(
        session_id=session_id,
        corpus_ref=corpus.corpus_ref,
        source_filename=corpus.source_filename,
        source_format=corpus.source_type.value,
        document_count=corpus.document_count,
        truncated=corpus.truncated,
        profile=profile_dict,
    )


@app.post("/query", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query(request: Request, payload: QueryRequest):
    # `request: Request` (Starlette's, not the request body) is required by @limiter.limit, which looks it
    # up by this exact parameter name — the JSON body is `payload` to avoid the name collision.
    structlog.contextvars.bind_contextvars(session_id=payload.session_id)
    try:
        session = session_store.get(payload.session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if session.corpus_ref is None:
        raise HTTPException(status_code=400, detail="No file has been uploaded for this session yet.")

    # run_agent is synchronous (it calls blocking HF/FAISS inference and Groq HTTP calls directly) --
    # awaiting it inline in this async def handler would hold the single event loop for its full duration,
    # serializing every concurrent /query request behind it (the exact bottleneck LOAD_TEST_RESULTS.md's
    # first run measured: ~1x effective concurrency across 15 simultaneous requests, not 10x). Offloading
    # to FastAPI's thread pool lets the event loop dispatch to other requests while this one runs, and lets
    # PyTorch/HTTP I/O actually overlap across threads instead of monopolizing the one thread that matters.
    result = await run_in_threadpool(
        run_agent,
        session_id=payload.session_id,
        corpus_ref=session.corpus_ref,
        user_query=payload.query,
        chat_history=session.chat_history,
        profile=session.profile,
    )

    if result.get("profile"):
        session_store.update_profile(payload.session_id, result["profile"])
    session_store.append_turn(payload.session_id, payload.query, result.get("final_answer"))

    return QueryResponse(
        session_id=payload.session_id,
        plan=result["plan"],
        tool_results=_serialize_tool_results(result["tool_results"]),
        final_answer=result.get("final_answer"),
        error=result.get("error"),
        latency=result["latency"],
    )


@app.get("/session/{session_id}/history", response_model=HistoryResponse)
async def history(session_id: str):
    structlog.contextvars.bind_contextvars(session_id=session_id)
    try:
        session = session_store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return HistoryResponse(
        session_id=session_id,
        chat_history=[HistoryTurn(**turn) for turn in session.chat_history],
    )


@app.get("/metrics")
async def metrics():
    """Basic in-process counters (item 3, 2026-09-02 scope revision) — request count, average latency,
    error rate. Not Prometheus-formatted; a plain JSON summary is the documented scope here (see
    docs/TECH_STACK.md)."""
    return metrics_registry.snapshot()
