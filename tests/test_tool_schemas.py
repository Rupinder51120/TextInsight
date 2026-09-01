import pytest
from pydantic import ValidationError

from tools.schemas import (
    ClassificationDocumentResult,
    ProfileDatasetInput,
    ProfileDatasetOutput,
    SentimentAnalysisInput,
    SentimentDocumentResult,
    SummarizeTextInput,
    TextClassificationInput,
)


class TestProfileDatasetSchemas:
    def test_input_requires_only_corpus_ref(self):
        input_ = ProfileDatasetInput(corpus_ref="abc123")
        assert input_.column_hint is None

    def test_output_allows_null_text_column_with_reason(self):
        output = ProfileDatasetOutput(
            n_documents=0,
            text_column=None,
            avg_length=None,
            length_distribution={},
            detected_language=None,
            has_labels=False,
            source_format="csv",
            reason="no usable text column",
        )
        assert output.text_column is None
        assert output.reason == "no usable text column"


class TestSentimentSchemas:
    def test_label_must_be_positive_or_negative(self):
        with pytest.raises(ValidationError):
            SentimentDocumentResult(id="0", label="neutral", score=0.5)

    def test_valid_label_accepted(self):
        result = SentimentDocumentResult(id="0", label="positive", score=0.9)
        assert result.label == "positive"

    def test_input_document_ids_optional(self):
        input_ = SentimentAnalysisInput(corpus_ref="abc")
        assert input_.document_ids is None


class TestTextClassificationSchemas:
    def test_candidate_labels_cannot_be_empty(self):
        with pytest.raises(ValidationError):
            TextClassificationInput(corpus_ref="abc", candidate_labels=[])

    def test_candidate_labels_with_values_accepted(self):
        input_ = TextClassificationInput(corpus_ref="abc", candidate_labels=["billing", "technical"])
        assert input_.candidate_labels == ["billing", "technical"]

    def test_output_result_carries_all_scores(self):
        result = ClassificationDocumentResult(
            id="0", label="billing", score=0.8, all_scores={"billing": 0.8, "technical": 0.2}
        )
        assert result.all_scores["billing"] == 0.8


class TestSummarizeTextSchemas:
    def test_mode_defaults_to_single(self):
        input_ = SummarizeTextInput(corpus_ref="abc")
        assert input_.mode == "single"

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValidationError):
            SummarizeTextInput(corpus_ref="abc", mode="not_a_real_mode")
