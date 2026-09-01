"""text_classification tool logic — mocked pipeline. Real-model correctness in test_tools_real_models.py."""

import sys
from unittest.mock import MagicMock

import pytest

from ingestion.corpus import Document
from ingestion.store import CorpusNotFoundError
from tests.helpers import register_corpus
from tools.text_classification import text_classification

# See tests/test_tools_sentiment_logic.py's comment: sys.modules is the only lookup immune to the
# tools/__init__.py re-export shadow, so monkeypatch targets must go through it, not a dotted string path.
text_classification_module = sys.modules["tools.text_classification"]


class TestTextClassificationLogic:
    def test_top_label_and_scores_mapped_correctly(self, monkeypatch):
        mock_pipe = MagicMock(
            return_value=[
                {"sequence": "x", "labels": ["billing", "technical"], "scores": [0.8, 0.2]},
            ]
        )
        monkeypatch.setattr(text_classification_module, "get_zero_shot_pipeline", lambda: mock_pipe)

        docs = [Document(id="0", text="My card was charged twice for one order.")]
        ref = register_corpus(docs)

        result = text_classification(corpus_ref=ref, candidate_labels=["billing", "technical"])

        assert len(result.per_document) == 1
        item = result.per_document[0]
        assert item.label == "billing"
        assert item.score == 0.8
        assert item.all_scores == {"billing": 0.8, "technical": 0.2}
        assert result.latency_ms >= 0

    def test_single_dict_result_normalized_to_list(self, monkeypatch):
        # HF's zero-shot pipeline returns a bare dict (not a list) for a single input text.
        mock_pipe = MagicMock(return_value={"sequence": "x", "labels": ["a", "b"], "scores": [0.6, 0.4]})
        monkeypatch.setattr(text_classification_module, "get_zero_shot_pipeline", lambda: mock_pipe)

        docs = [Document(id="0", text="A single document to classify.")]
        ref = register_corpus(docs)

        result = text_classification(corpus_ref=ref, candidate_labels=["a", "b"])

        assert len(result.per_document) == 1
        assert result.per_document[0].label == "a"

    def test_empty_candidate_labels_raises(self):
        docs = [Document(id="0", text="some text")]
        ref = register_corpus(docs)

        with pytest.raises(ValueError):
            text_classification(corpus_ref=ref, candidate_labels=[])

    def test_empty_documents_skipped_without_calling_pipeline(self, monkeypatch):
        mock_pipe = MagicMock()
        monkeypatch.setattr(text_classification_module, "get_zero_shot_pipeline", lambda: mock_pipe)

        docs = [Document(id="0", text="   ")]
        ref = register_corpus(docs)

        result = text_classification(corpus_ref=ref, candidate_labels=["a", "b"])

        assert result.per_document == []
        mock_pipe.assert_not_called()

    def test_unknown_corpus_ref_raises(self):
        with pytest.raises(CorpusNotFoundError):
            text_classification(corpus_ref="does-not-exist", candidate_labels=["a"])
