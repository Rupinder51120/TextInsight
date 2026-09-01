from ingestion.corpus import Corpus, Document, SourceType, make_corpus_ref
from ingestion.store import corpus_store


def register_corpus(
    documents: list[Document],
    source_type: SourceType = SourceType.CSV,
    filename: str = "test.csv",
) -> str:
    """Build a Corpus from raw Documents and register it in the shared corpus_store, returning its ref."""
    corpus = Corpus(
        corpus_ref=make_corpus_ref(),
        source_type=source_type,
        source_filename=filename,
        documents=documents,
    )
    corpus_store.put(corpus)
    return corpus.corpus_ref
