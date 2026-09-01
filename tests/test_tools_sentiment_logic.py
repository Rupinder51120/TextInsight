"""Sentiment tool logic — mocked pipeline, no model download (fast). Real-model correctness is in
tests/test_tools_real_models.py per docs/TESTING_STRATEGY.md §1 ("mocked model outputs where model loading
itself is slow; real-model smoke tests kept separate, see §4")."""

import sys
from unittest.mock import MagicMock

import pytest

from ingestion.corpus import Document
from ingestion.store import CorpusNotFoundError
from tests.helpers import register_corpus
from tools.sentiment_analysis import sentiment_analysis

# tools/__init__.py re-exports `sentiment_analysis` under the same name as this submodule, which shadows
# `tools.sentiment_analysis` (the module) with the function on the `tools` package object — even
# `import tools.sentiment_analysis as x` resolves through that shadowed attribute. sys.modules is the only
# lookup immune to it, so monkeypatch targets must go through that instead of a dotted string path.
sentiment_analysis_module = sys.modules["tools.sentiment_analysis"]


class TestSentimentAnalysisLogic:
    def test_skips_too_short_documents(self, monkeypatch):
        mock_pipe = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.99}])
        monkeypatch.setattr(sentiment_analysis_module, "get_sentiment_pipeline", lambda: mock_pipe)

        docs = [
            Document(id="0", text="This is a wonderfully long positive review of the product."),
            Document(id="1", text="ok"),
        ]
        ref = register_corpus(docs)

        result = sentiment_analysis(corpus_ref=ref)

        assert result.skipped_count == 1
        assert len(result.per_document) == 1
        assert result.per_document[0].label == "positive"
        assert result.distribution == {"positive": 1.0, "negative": 0.0}
        assert result.latency_ms >= 0

    def test_document_ids_filter_only_calls_pipeline_on_subset(self, monkeypatch):
        mock_pipe = MagicMock(return_value=[{"label": "NEGATIVE", "score": 0.9}])
        monkeypatch.setattr(sentiment_analysis_module, "get_sentiment_pipeline", lambda: mock_pipe)

        docs = [
            Document(id="0", text="A decent length document one here."),
            Document(id="1", text="A decent length document two here."),
        ]
        ref = register_corpus(docs)

        result = sentiment_analysis(corpus_ref=ref, document_ids=["1"])

        assert len(result.per_document) == 1
        assert result.per_document[0].id == "1"
        mock_pipe.assert_called_once()
        called_texts = mock_pipe.call_args[0][0]
        assert called_texts == ["A decent length document two here."]

    def test_all_documents_skipped_returns_empty_result_without_calling_pipeline(self, monkeypatch):
        mock_pipe = MagicMock()
        monkeypatch.setattr(sentiment_analysis_module, "get_sentiment_pipeline", lambda: mock_pipe)

        docs = [Document(id="0", text="a"), Document(id="1", text="")]
        ref = register_corpus(docs)

        result = sentiment_analysis(corpus_ref=ref)

        assert result.per_document == []
        assert result.skipped_count == 2
        assert result.distribution == {}
        mock_pipe.assert_not_called()

    def test_text_column_override_reads_from_metadata(self, monkeypatch):
        mock_pipe = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.8}])
        monkeypatch.setattr(sentiment_analysis_module, "get_sentiment_pipeline", lambda: mock_pipe)

        docs = [Document(id="0", text="fallback text", metadata={"title": "A genuinely great title here"})]
        ref = register_corpus(docs)

        sentiment_analysis(corpus_ref=ref, text_column="title")

        called_texts = mock_pipe.call_args[0][0]
        assert called_texts == ["A genuinely great title here"]

    def test_text_column_matching_default_ingestion_column_uses_doc_text(self, monkeypatch):
        # Regression: profile_dataset's default text_column is the *name* of the column ingestion already
        # used to build doc.text (e.g. "review_text"), which is deliberately absent from doc.metadata
        # (it's already doc.text). Passing that name straight through as text_column must not cause every
        # document to look empty.
        mock_pipe = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.8}])
        monkeypatch.setattr(sentiment_analysis_module, "get_sentiment_pipeline", lambda: mock_pipe)

        docs = [
            Document(
                id="0",
                text="This product completely changed how I work.",
                metadata={"text_column": "review_text", "rating": 5},
            )
        ]
        ref = register_corpus(docs)

        result = sentiment_analysis(corpus_ref=ref, text_column="review_text")

        assert result.skipped_count == 0
        called_texts = mock_pipe.call_args[0][0]
        assert called_texts == ["This product completely changed how I work."]

    def test_unknown_corpus_ref_raises(self):
        with pytest.raises(CorpusNotFoundError):
            sentiment_analysis(corpus_ref="does-not-exist")
