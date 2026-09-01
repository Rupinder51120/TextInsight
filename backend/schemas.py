"""FastAPI request/response schemas — docs/ARCHITECTURE.md §2, §4. Distinct from tools/schemas.py, which
covers tool I/O; these are the HTTP-layer contracts for /upload, /query, /session/{id}/history.
"""

from typing import Any

from pydantic import BaseModel


class UploadResponse(BaseModel):
    session_id: str
    corpus_ref: str
    source_filename: str
    source_format: str
    document_count: int
    truncated: bool
    profile: dict[str, Any]


class QueryRequest(BaseModel):
    session_id: str
    query: str


class QueryResponse(BaseModel):
    session_id: str
    plan: list[str]
    tool_results: dict[str, Any]
    final_answer: str | None
    error: str | None
    latency: dict[str, float]


class HistoryTurn(BaseModel):
    role: str
    content: str | None


class HistoryResponse(BaseModel):
    session_id: str
    chat_history: list[HistoryTurn]
