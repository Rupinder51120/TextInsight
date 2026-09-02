"""FAISS index build/query + on-disk persistence, one index per corpus_ref.

Per docs/TOOLS_AND_MODELS.md #6-7 and docs/ARCHITECTURE.md §2/§9: "one index per uploaded corpus
(session-scoped)... persisted to disk per session directory so repeated queries don't re-embed."

Versioning: each upload gets a fresh, unique corpus_ref (ingestion/corpus.py's make_corpus_ref) — so
corpus_ref uniqueness itself already encodes "corpus version" (matches docs/DATA_FLOW.md §3: a new upload
gets a new corpus_ref, naturally invalidating any old index rather than colliding with it). An index existing
on disk for a given corpus_ref is therefore sufficient to treat it as valid/cached.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim) — the same default used for both indexing and
query-time embedding, per docs/TOOLS_AND_MODELS.md #7 ("same embedding model as generate_embeddings").
"""

import json
import threading
from pathlib import Path

import numpy as np

from ingestion.corpus import Document

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_INDEX_DIR = Path(__file__).resolve().parent.parent / "sessions" / "faiss"
_model_cache: dict[str, object] = {}
_model_lock = threading.Lock()


def _get_embedding_model():
    """Double-checked locking, same pattern as models/registry.py's get_pipeline(). This alone turned out
    NOT to be sufficient once backend/main.py started offloading /query's agent execution to a thread
    pool: concurrent first-time calls from multiple threads into this function AND models/registry.py's
    get_pipeline() (two different locks, two different caches) could still race on transformers' own
    shared internal lazy-import state, observed as "cannot import name 'AutoConfig' from 'transformers'"
    and "'module' object is not callable" under concurrent load — sometimes crashing the process outright.
    backend/main.py's startup warm-up (warm_up_all_models) is the real fix: every model, including this
    one, is loaded once before the server accepts any traffic, so this function's cache is always already
    populated by the time a second thread could ever reach it concurrently. The lock stays as a defense-in-
    depth guard, not the primary fix."""
    if "model" not in _model_cache:
        with _model_lock:
            if "model" not in _model_cache:
                from sentence_transformers import SentenceTransformer

                _model_cache["model"] = SentenceTransformer(EMBEDDING_MODEL)
    return _model_cache["model"]


def warm_up_embedding_model() -> None:
    """Eagerly load the embedding model — called once at process startup (backend/main.py's lifespan), not
    per-request. See _get_embedding_model's docstring for why this, not just locking, is the real fix for
    the concurrent-first-load race."""
    _get_embedding_model()


def _paths(corpus_ref: str) -> tuple[Path, Path]:
    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    return _INDEX_DIR / f"{corpus_ref}.index", _INDEX_DIR / f"{corpus_ref}.ids.json"


def index_exists(corpus_ref: str) -> bool:
    index_path, ids_path = _paths(corpus_ref)
    return index_path.exists() and ids_path.exists()


def build_index(corpus_ref: str, documents: list[Document]) -> tuple[int, int]:
    """Embeds and indexes all non-empty documents. Returns (n_vectors, dim).

    Raises ValueError if there are no usable (non-empty) texts — docs/TOOLS_AND_MODELS.md #6: "corpus too
    small (e.g., 0 usable texts) → explicit error, no empty index silently created."
    """
    import faiss

    usable = [d for d in documents if d.text.strip()]
    if not usable:
        raise ValueError("generate_embeddings: no usable (non-empty) texts to index in this corpus.")

    model = _get_embedding_model()
    vectors = model.encode([d.text for d in usable], convert_to_numpy=True, normalize_embeddings=True)
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine similarity via inner product on normalized vectors
    index.add(vectors.astype(np.float32))

    index_path, ids_path = _paths(corpus_ref)
    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps([d.id for d in usable]))

    return len(usable), dim


def _load_index(corpus_ref: str):
    import faiss

    index_path, ids_path = _paths(corpus_ref)
    index = faiss.read_index(str(index_path))
    ids = json.loads(ids_path.read_text())
    return index, ids


def index_stats(corpus_ref: str) -> tuple[int, int]:
    """Returns (n_vectors, dim) for an already-built index, without re-embedding."""
    index, ids = _load_index(corpus_ref)
    return len(ids), index.d


def query_index(corpus_ref: str, query_text: str, top_k: int) -> list[tuple[str, float]]:
    """Returns up to top_k (document_id, score) pairs, ranked by similarity, highest first."""
    index, ids = _load_index(corpus_ref)
    model = _get_embedding_model()
    vector = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    scores, indices = index.search(vector, min(top_k, len(ids)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append((ids[idx], float(score)))
    return results
