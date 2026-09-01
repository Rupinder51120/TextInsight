"""semantic_search — docs/TOOLS_AND_MODELS.md #7.

Requires an existing FAISS index; per the doc, auto-triggering generate_embeddings when missing is
agent-level chaining, "not hidden inside this tool" — so a missing index is a clear error here, not a
silent auto-build.
"""

from ingestion.store import corpus_store
from models.faiss_index import index_exists, query_index
from tools.schemas import SemanticSearchOutput, SemanticSearchResult
from tools.timing import timed_tool

_EXCERPT_CHARS = 280


@timed_tool
def semantic_search(corpus_ref: str, query: str, top_k: int = 5) -> SemanticSearchOutput:
    corpus = corpus_store.get(corpus_ref)
    if not index_exists(corpus_ref):
        raise ValueError(
            f"semantic_search: no embeddings index exists yet for corpus_ref={corpus_ref!r} — "
            "run generate_embeddings first."
        )

    docs_by_id = {d.id: d for d in corpus.documents}
    hits = query_index(corpus_ref, query, top_k)

    results = [
        SemanticSearchResult(id=doc_id, text_excerpt=docs_by_id[doc_id].text[:_EXCERPT_CHARS], score=score)
        for doc_id, score in hits
        if doc_id in docs_by_id
    ]
    return SemanticSearchOutput(results=results)
