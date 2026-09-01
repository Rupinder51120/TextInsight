"""summarize_text tool logic — mocked pipeline. Real-model correctness in test_tools_real_models.py."""

import sys
from unittest.mock import MagicMock

import pytest

from ingestion.corpus import Document
from ingestion.store import CorpusNotFoundError
from tests.helpers import register_corpus
from tools.summarize_text import summarize_text

# See tests/test_tools_sentiment_logic.py's comment: sys.modules is the only lookup immune to the
# tools/__init__.py re-export shadow, so monkeypatch targets must go through it, not a dotted string path.
summarize_text_module = sys.modules["tools.summarize_text"]


def _mock_pipe():
    return MagicMock(side_effect=lambda text, **kwargs: [{"summary_text": f"SUMMARY({len(text)} chars)"}])


class TestSummarizeTextLogic:
    def test_single_mode_summarizes_one_document(self, monkeypatch):
        mock_pipe = _mock_pipe()
        monkeypatch.setattr(summarize_text_module, "get_summarization_pipeline", lambda: mock_pipe)

        docs = [Document(id="0", text="This is a short document about a product review.")]
        ref = register_corpus(docs)

        result = summarize_text(corpus_ref=ref, document_ids=["0"], mode="single")

        assert result.source_document_ids == ["0"]
        assert result.chunked is False
        assert "SUMMARY" in result.summary
        assert result.latency_ms >= 0
        mock_pipe.assert_called_once()

    def test_single_mode_with_multiple_documents_raises(self, monkeypatch):
        monkeypatch.setattr(summarize_text_module, "get_summarization_pipeline", _mock_pipe)

        docs = [Document(id="0", text="doc one"), Document(id="1", text="doc two")]
        ref = register_corpus(docs)

        with pytest.raises(ValueError):
            summarize_text(corpus_ref=ref, mode="single")

    def test_batch_digest_concatenates_documents(self, monkeypatch):
        mock_pipe = _mock_pipe()
        monkeypatch.setattr(summarize_text_module, "get_summarization_pipeline", lambda: mock_pipe)

        docs = [Document(id="0", text="First review."), Document(id="1", text="Second review.")]
        ref = register_corpus(docs)

        result = summarize_text(corpus_ref=ref, mode="batch_digest")

        assert set(result.source_document_ids) == {"0", "1"}
        assert result.chunked is False
        mock_pipe.assert_called_once()

    def test_long_input_is_chunked_map_reduce(self, monkeypatch):
        mock_pipe = _mock_pipe()
        monkeypatch.setattr(summarize_text_module, "get_summarization_pipeline", lambda: mock_pipe)

        long_text = "lorem ipsum " * 300  # 3600 chars > _MAX_CHUNK_CHARS (3000) -> 2 chunks
        docs = [Document(id="0", text=long_text)]
        ref = register_corpus(docs)

        result = summarize_text(corpus_ref=ref, document_ids=["0"], mode="single")

        assert result.chunked is True
        assert mock_pipe.call_count == 2

    def test_empty_selection_raises(self, monkeypatch):
        monkeypatch.setattr(summarize_text_module, "get_summarization_pipeline", _mock_pipe)

        docs = [Document(id="0", text="   ")]
        ref = register_corpus(docs)

        with pytest.raises(ValueError):
            summarize_text(corpus_ref=ref, mode="single")

    def test_unknown_mode_raises(self, monkeypatch):
        monkeypatch.setattr(summarize_text_module, "get_summarization_pipeline", _mock_pipe)

        docs = [Document(id="0", text="some text here")]
        ref = register_corpus(docs)

        with pytest.raises(ValueError):
            summarize_text(corpus_ref=ref, mode="not_a_mode")

    def test_unknown_corpus_ref_raises(self):
        with pytest.raises(CorpusNotFoundError):
            summarize_text(corpus_ref="does-not-exist")
