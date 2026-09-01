"""summarize_text — docs/TOOLS_AND_MODELS.md #5.

Model: sshleifer/distilbart-cnn-12-6. Input exceeding the model's practical context is chunked and
summarized map-then-reduce (docs/TOOLS_AND_MODELS.md: "chunking behavior surfaced in output metadata" —
here via the `chunked` output field); the character-based chunk size (_MAX_CHUNK_CHARS) is an explicit
assumption approximating the model's ~1024-token limit, since the docs don't pin an exact figure
(CLAUDE.md §6).
"""

from ingestion.corpus import Document
from ingestion.store import corpus_store
from models.registry import get_summarization_pipeline
from tools.schemas import SummarizeTextOutput
from tools.timing import timed_tool

_MAX_CHUNK_CHARS = 3000


def _select_documents(corpus_documents: list[Document], document_ids: list[str] | None) -> list[Document]:
    if document_ids is None:
        return corpus_documents
    wanted = set(document_ids)
    return [doc for doc in corpus_documents if doc.id in wanted]


def _chunk_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + max_chars])
        start += max_chars
    return chunks


def _summarize_chunk(pipe, text: str) -> str:
    input_len = len(text.split())
    max_length = max(10, min(130, input_len // 2 or 10))
    min_length = max(5, min(30, max_length // 2))
    result = pipe(text, max_length=max_length, min_length=min_length, truncation=True, do_sample=False)
    return result[0]["summary_text"].strip()


@timed_tool
def summarize_text(
    corpus_ref: str,
    document_ids: list[str] | None = None,
    mode: str = "single",
) -> SummarizeTextOutput:
    corpus = corpus_store.get(corpus_ref)
    documents = _select_documents(corpus.documents, document_ids)
    documents = [doc for doc in documents if doc.text.strip()]

    if not documents:
        raise ValueError("summarize_text: no non-empty documents to summarize for the given selection.")

    if mode == "single":
        if len(documents) != 1:
            raise ValueError(
                f"summarize_text mode='single' requires exactly one document, got {len(documents)}. "
                "Use mode='batch_digest' for multiple documents."
            )
        target_text = documents[0].text
        source_ids = [documents[0].id]
    elif mode == "batch_digest":
        target_text = "\n\n".join(doc.text for doc in documents)
        source_ids = [doc.id for doc in documents]
    else:
        raise ValueError(f"summarize_text: unknown mode '{mode}' (expected 'single' or 'batch_digest').")

    pipe = get_summarization_pipeline()
    chunks = _chunk_text(target_text, _MAX_CHUNK_CHARS)
    chunked = len(chunks) > 1

    chunk_summaries = [_summarize_chunk(pipe, chunk) for chunk in chunks]

    if len(chunk_summaries) == 1:
        final_summary = chunk_summaries[0]
    else:
        combined = " ".join(chunk_summaries)
        final_summary = _summarize_chunk(pipe, combined) if len(combined) > _MAX_CHUNK_CHARS else combined

    return SummarizeTextOutput(summary=final_summary, source_document_ids=source_ids, chunked=chunked)
