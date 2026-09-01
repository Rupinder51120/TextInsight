"""Shared Pydantic input/output schemas for the NLP tools (Days 2-3).

Field names/shapes match docs/TOOLS_AND_MODELS.md's per-tool Input/Output contracts. Every Output model
carries `latency_ms` (populated by @timed_tool, tools/timing.py) so a tool is independently timed even when
called outside the agent graph — docs/LATENCY_AND_PERFORMANCE.md §3 requires timing on every tool;
`execute_tool` (agent/nodes.py) folds this value into `AgentState.latency` when the tool runs inside a turn.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# profile_dataset
# ---------------------------------------------------------------------------


class ProfileDatasetInput(BaseModel):
    corpus_ref: str
    column_hint: str | None = None


class ProfileDatasetOutput(BaseModel):
    n_documents: int
    text_column: str | None
    avg_length: float | None
    length_distribution: dict[str, float]
    detected_language: str | None
    has_labels: bool
    label_column: str | None = None
    class_distribution: dict[str, float] | None = None
    source_format: str
    reason: str | None = None
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# sentiment_analysis
# ---------------------------------------------------------------------------


class SentimentAnalysisInput(BaseModel):
    corpus_ref: str
    document_ids: list[str] | None = None
    text_column: str | None = None


class SentimentDocumentResult(BaseModel):
    id: str
    label: Literal["positive", "negative"]
    score: float


class SentimentAnalysisOutput(BaseModel):
    per_document: list[SentimentDocumentResult]
    distribution: dict[str, float]
    skipped_count: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# text_classification (zero-shot)
# ---------------------------------------------------------------------------


class TextClassificationInput(BaseModel):
    corpus_ref: str
    document_ids: list[str] | None = None
    candidate_labels: list[str] = Field(min_length=1)


class ClassificationDocumentResult(BaseModel):
    id: str
    label: str
    score: float
    all_scores: dict[str, float]


class TextClassificationOutput(BaseModel):
    per_document: list[ClassificationDocumentResult]
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# summarize_text
# ---------------------------------------------------------------------------


class SummarizeTextInput(BaseModel):
    corpus_ref: str
    document_ids: list[str] | None = None
    mode: Literal["single", "batch_digest"] = "single"


class SummarizeTextOutput(BaseModel):
    summary: str
    source_document_ids: list[str]
    chunked: bool = False
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# generate_embeddings
# ---------------------------------------------------------------------------


class GenerateEmbeddingsInput(BaseModel):
    corpus_ref: str


class GenerateEmbeddingsOutput(BaseModel):
    index_id: str
    n_vectors: int
    dim: int
    built: bool
    cached: bool
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# semantic_search
# ---------------------------------------------------------------------------


class SemanticSearchInput(BaseModel):
    corpus_ref: str
    query: str
    top_k: int = 5


class SemanticSearchResult(BaseModel):
    id: str
    text_excerpt: str
    score: float


class SemanticSearchOutput(BaseModel):
    results: list[SemanticSearchResult]
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# filter_documents
# ---------------------------------------------------------------------------


class FilterDocumentsInput(BaseModel):
    """Public contract per docs/TOOLS_AND_MODELS.md #8. The actual Python function additionally takes
    `source_result` (the referenced prior tool's real output) — internal plumbing that execute_tool
    supplies from AgentState.tool_results, never part of the documented/LLM-facing input schema."""

    corpus_ref: str
    criteria: dict[str, Any]


class FilterDocumentsOutput(BaseModel):
    document_ids: list[str]
    count: int
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# model_recommendation — shared rule-engine types (docs/MODEL_RECOMMENDATION.md §2-3)
# ---------------------------------------------------------------------------


class UserConstraints(BaseModel):
    latency_requirement: str | None = None  # e.g. "fast", "real-time"; None = unspecified
    compute_constraints: str | None = None  # e.g. "cpu_only", "gpu_available"; None = default cpu_only


class CandidateModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # "model_name" is the doc-specified field name

    model_name: str
    is_default: bool
    reason: str


# ---------------------------------------------------------------------------
# evaluate_candidates — docs/TOOLS_AND_MODELS.md #10a
# ---------------------------------------------------------------------------


class EvaluateCandidatesInput(BaseModel):
    corpus_ref: str
    profile: dict[str, Any]
    candidate_models: list[str]
    sample_size: int = 500


class CandidateEvaluation(BaseModel):
    model_config = ConfigDict(protected_namespaces=())  # "model_name" is the doc-specified field name

    model_name: str
    accuracy: float
    f1: float
    n_examples: int


class EvaluateCandidatesOutput(BaseModel):
    per_model: list[CandidateEvaluation]
    skipped: bool
    skip_reason: str | None = None
    latency_ms: float = 0.0
