"""filter_documents — deterministic, no model involved. docs/TOOLS_AND_MODELS.md #8."""

import pytest

from ingestion.corpus import Document
from ingestion.store import CorpusNotFoundError
from tests.helpers import register_corpus
from tools.filter_documents import filter_documents
from tools.schemas import (
    SemanticSearchOutput,
    SemanticSearchResult,
    SentimentAnalysisOutput,
    SentimentDocumentResult,
)


class TestFilterDocuments:
    def test_filters_negative_sentiment_documents(self):
        docs = [Document(id=str(i), text=f"doc {i}") for i in range(4)]
        ref = register_corpus(docs)
        sentiment_result = SentimentAnalysisOutput(
            per_document=[
                SentimentDocumentResult(id="0", label="positive", score=0.9),
                SentimentDocumentResult(id="1", label="negative", score=0.8),
                SentimentDocumentResult(id="2", label="negative", score=0.7),
                SentimentDocumentResult(id="3", label="positive", score=0.95),
            ],
            distribution={"positive": 0.5, "negative": 0.5},
        )

        result = filter_documents(
            corpus_ref=ref,
            criteria={"from_tool": "sentiment_analysis", "field": "label", "equals": "negative"},
            source_result=sentiment_result,
        )

        assert result.document_ids == ["1", "2"]
        assert result.count == 2
        assert result.latency_ms >= 0

    def test_filters_top_k_from_semantic_search(self):
        docs = [Document(id=str(i), text=f"doc {i}") for i in range(5)]
        ref = register_corpus(docs)
        search_result = SemanticSearchOutput(
            results=[
                SemanticSearchResult(id="2", text_excerpt="x", score=0.9),
                SemanticSearchResult(id="0", text_excerpt="x", score=0.8),
                SemanticSearchResult(id="4", text_excerpt="x", score=0.5),
            ]
        )

        result = filter_documents(
            corpus_ref=ref,
            criteria={"from_tool": "semantic_search", "top_k": 2},
            source_result=search_result,
        )

        assert result.document_ids == ["2", "0"]
        assert result.count == 2

    def test_missing_source_result_raises(self):
        docs = [Document(id="0", text="doc")]
        ref = register_corpus(docs)

        with pytest.raises(ValueError):
            filter_documents(
                corpus_ref=ref,
                criteria={"from_tool": "sentiment_analysis", "field": "label", "equals": "negative"},
                source_result=None,
            )

    def test_unsupported_result_type_raises(self):
        docs = [Document(id="0", text="doc")]
        ref = register_corpus(docs)

        with pytest.raises(ValueError):
            filter_documents(corpus_ref=ref, criteria={"from_tool": "profile_dataset"}, source_result=object())

    def test_unknown_corpus_ref_raises(self):
        with pytest.raises(CorpusNotFoundError):
            filter_documents(corpus_ref="does-not-exist", criteria={}, source_result=None)
