"""generate_embeddings — docs/TOOLS_AND_MODELS.md #6. Idempotent: a second call for the same corpus_ref
returns cached: true without recomputation (see models/faiss_index.py's versioning note)."""

from ingestion.store import corpus_store
from models.faiss_index import build_index, index_exists, index_stats
from tools.schemas import GenerateEmbeddingsOutput
from tools.timing import timed_tool


@timed_tool
def generate_embeddings(corpus_ref: str) -> GenerateEmbeddingsOutput:
    corpus_store.get(corpus_ref)  # validates corpus_ref exists

    if index_exists(corpus_ref):
        n_vectors, dim = index_stats(corpus_ref)
        return GenerateEmbeddingsOutput(index_id=corpus_ref, n_vectors=n_vectors, dim=dim, built=False, cached=True)

    corpus = corpus_store.get(corpus_ref)
    n_vectors, dim = build_index(corpus_ref, corpus.documents)
    return GenerateEmbeddingsOutput(index_id=corpus_ref, n_vectors=n_vectors, dim=dim, built=True, cached=False)
