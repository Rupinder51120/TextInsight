"""FastAPI backend — docs/ARCHITECTURE.md §2, §4. Thin API layer: owns session lifecycle and request
validation (Pydantic), invokes the LangGraph agent; no business logic beyond that, per the doc.
"""

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
from tools.profile_dataset import profile_dataset

app = FastAPI(title="TextInsight API")

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
    try:
        session = session_store.get(session_id)
    except SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return HistoryResponse(
        session_id=session_id,
        chat_history=[HistoryTurn(**turn) for turn in session.chat_history],
    )
