"""generate_embeddings — docs/TESTING_STRATEGY.md §6: idempotency + invalidation-on-new-corpus. Uses the
real (small, fast) embedding model rather than mocking, since it's cheap and correctness of the FAISS
build/cache mechanics is exactly what's under test."""

import pytest

from ingestion.corpus import Document
from ingestion.store import CorpusNotFoundError
from tests.helpers import register_corpus
from tools.generate_embeddings import generate_embeddings


class TestGenerateEmbeddings:
    def test_first_call_builds_index(self):
        docs = [Document(id=str(i), text=f"Some review text number {i} about a product.") for i in range(5)]
        ref = register_corpus(docs)

        result = generate_embeddings(corpus_ref=ref)

        assert result.built is True
        assert result.cached is False
        assert result.n_vectors == 5
        assert result.dim == 384
        assert result.latency_ms >= 0

    def test_second_call_is_cached_and_not_rebuilt(self):
        docs = [Document(id=str(i), text=f"Some review text number {i} about a product.") for i in range(5)]
        ref = register_corpus(docs)

        first = generate_embeddings(corpus_ref=ref)
        second = generate_embeddings(corpus_ref=ref)

        assert first.built is True and first.cached is False
        assert second.built is False and second.cached is True
        assert second.n_vectors == first.n_vectors
        assert second.dim == first.dim

    def test_new_corpus_gets_its_own_independent_index(self):
        docs_a = [Document(id=str(i), text=f"Corpus A document {i}.") for i in range(3)]
        docs_b = [Document(id=str(i), text=f"Corpus B document {i}.") for i in range(6)]
        ref_a = register_corpus(docs_a)
        ref_b = register_corpus(docs_b)

        result_a = generate_embeddings(corpus_ref=ref_a)
        result_b = generate_embeddings(corpus_ref=ref_b)

        assert result_a.cached is False
        assert result_b.cached is False
        assert result_a.n_vectors == 3
        assert result_b.n_vectors == 6

    def test_empty_corpus_raises(self):
        docs = [Document(id="0", text="   ")]
        ref = register_corpus(docs)

        with pytest.raises(ValueError):
            generate_embeddings(corpus_ref=ref)

    def test_unknown_corpus_ref_raises(self):
        with pytest.raises(CorpusNotFoundError):
            generate_embeddings(corpus_ref="does-not-exist")
