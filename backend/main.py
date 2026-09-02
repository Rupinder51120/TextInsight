"""FastAPI backend — docs/ARCHITECTURE.md §2, §4. Thin API layer: owns session lifecycle and request
validation (Pydantic), invokes the LangGraph agent; no business logic beyond that, per the doc.
"""

import time

import structlog
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

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
from observability.logging import get_logger
from observability.metrics import metrics_registry
from tools.profile_dataset import profile_dataset

app = FastAPI(title="TextInsight API")

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
async def query(request: QueryRequest):
    structlog.contextvars.bind_contextvars(session_id=request.session_id)
    try:
        session = session_store.get(request.session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if session.corpus_ref is None:
        raise HTTPException(status_code=400, detail="No file has been uploaded for this session yet.")

    result = run_agent(
        session_id=request.session_id,
        corpus_ref=session.corpus_ref,
        user_query=request.query,
        chat_history=session.chat_history,
        profile=session.profile,
    )

    if result.get("profile"):
        session_store.update_profile(request.session_id, result["profile"])
    session_store.append_turn(request.session_id, request.query, result.get("final_answer"))

    return QueryResponse(
        session_id=request.session_id,
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
