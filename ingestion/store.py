"""CorpusStore — the single encapsulated point where a corpus_ref resolves to real Corpus data.

Every tool's documented input contract (docs/TOOLS_AND_MODELS.md) is `corpus_ref`, not a raw Corpus object —
tools must be able to resolve one from the other to be independently callable/testable per
docs/ARCHITECTURE.md §2. This is intentionally NOT the module-level-`df`-many-tools-mutate-directly pattern
CLAUDE.md §4 warns against: it's a small encapsulated class with a defined get/put contract, in-process only,
matching CLAUDE.md §3.5 ("state lives in-process, in AgentState / session-scoped memory, keyed by
session_id"). Full session lifecycle (per-session keying, upload-invalidates-previous-corpus notices) is
FastAPI's job on Day 5 (docs/ARCHITECTURE.md §2) — this class only provides the storage primitive that work
will sit on top of.
"""

from ingestion.corpus import Corpus


class CorpusNotFoundError(KeyError):
    pass


class CorpusStore:
    def __init__(self) -> None:
        self._corpora: dict[str, Corpus] = {}

    def put(self, corpus: Corpus) -> str:
        self._corpora[corpus.corpus_ref] = corpus
        return corpus.corpus_ref

    def get(self, corpus_ref: str) -> Corpus:
        try:
            return self._corpora[corpus_ref]
        except KeyError:
            raise CorpusNotFoundError(f"No corpus found for corpus_ref={corpus_ref!r}") from None


corpus_store = CorpusStore()
