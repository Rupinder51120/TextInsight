"""semantic_search — docs/TESTING_STRATEGY.md §6: query correctness on a small fixture with an obviously
relevant and an obviously irrelevant document."""

import pytest

from ingestion.corpus import Document
from ingestion.store import CorpusNotFoundError
from tests.helpers import register_corpus
from tools.generate_embeddings import generate_embeddings
from tools.semantic_search import semantic_search


class TestSemanticSearch:
    def test_ranks_relevant_document_above_irrelevant_one(self):
        docs = [
            Document(id="relevant", text="The delivery was delayed by two weeks and no one informed us."),
            Document(id="irrelevant", text="I love the color and the packaging design of this item."),
        ]
        ref = register_corpus(docs)
        generate_embeddings(corpus_ref=ref)

        result = semantic_search(corpus_ref=ref, query="complaints about delayed delivery", top_k=2)

        assert result.results[0].id == "relevant"
        assert result.results[0].score > result.results[1].score
        assert result.latency_ms >= 0

    def test_top_k_limits_results(self):
        docs = [Document(id=str(i), text=f"Document number {i} about something generic.") for i in range(10)]
        ref = register_corpus(docs)
        generate_embeddings(corpus_ref=ref)

        result = semantic_search(corpus_ref=ref, query="something generic", top_k=3)

        assert len(result.results) == 3

    def test_missing_index_raises_clear_error(self):
        docs = [Document(id="0", text="never indexed")]
        ref = register_corpus(docs)

        with pytest.raises(ValueError):
            semantic_search(corpus_ref=ref, query="anything", top_k=5)

    def test_unknown_corpus_ref_raises(self):
        with pytest.raises(CorpusNotFoundError):
            semantic_search(corpus_ref="does-not-exist", query="anything")
