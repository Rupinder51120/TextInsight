"""model_recommendation — docs/TESTING_STRATEGY.md §7. Mocked LLM for deterministic, fast runs. The A/B/C
schema-separation checks here are structural (independent of LLM prose content), per §7's explicit
requirement. Live end-to-end coverage is in tests/test_tools_model_recommendation_live.py.
"""

import json
import sys

import pytest

from tools.model_recommendation import model_recommendation
from tools.schemas import CandidateEvaluation, EvaluateCandidatesOutput, ResearchEvidence, UserConstraints

# See tests/test_tools_sentiment_logic.py's comment: sys.modules is the only lookup immune to a future
# tools/__init__.py re-export shadow, so monkeypatch targets go through it.
model_recommendation_module = sys.modules["tools.model_recommendation"]


def _mock_llm(monkeypatch, response: dict):
    from unittest.mock import MagicMock

    mock_client = MagicMock()
    mock_client.complete.return_value = json.dumps(response)
    monkeypatch.setattr(model_recommendation_module, "GroqLLMClient", lambda: mock_client)
    return mock_client


_GOOD_LLM_RESPONSE = {
    "recommendation": "distilbert-base-uncased-finetuned-sst-2-english",
    "rationale": ["Small dataset favors a pretrained default.", "Fast on CPU."],
    "system_judgment": "Recommended based on dataset characteristics and stated constraints: DistilBERT-SST2.",
    "confidence_note": "Moderate confidence given dataset size.",
}


class TestSmallUnlabeledDataset:
    def test_section_a_explicitly_states_no_evaluation_ran(self, monkeypatch):
        _mock_llm(monkeypatch, _GOOD_LLM_RESPONSE)

        result = model_recommendation(
            profile={"has_labels": False, "n_documents": 15},
            task_type="sentiment",
        )

        assert result.measured_on_user_data == []
        assert result.measured_skip_reason is not None
        assert "no labels" in result.measured_skip_reason.lower() or "no evaluation" in result.measured_skip_reason.lower()

    def test_disclaimer_present_regardless_of_llm_prose(self, monkeypatch):
        _mock_llm(monkeypatch, _GOOD_LLM_RESPONSE)

        result = model_recommendation(profile={"has_labels": False, "n_documents": 15}, task_type="sentiment")

        assert "does not perform training or fine-tuning" in result.fine_tune_note


class TestLabeledDatasetWithEvaluation:
    def test_measured_numbers_flow_through_unmodified_end_to_end(self, monkeypatch):
        _mock_llm(monkeypatch, _GOOD_LLM_RESPONSE)

        evaluation = EvaluateCandidatesOutput(
            per_model=[
                CandidateEvaluation(model_name="distilbert-base-uncased-finetuned-sst-2-english", accuracy=0.92, f1=0.91, n_examples=25),
                CandidateEvaluation(model_name="textattack/bert-base-uncased-SST-2", accuracy=0.6, f1=0.55, n_examples=25),
            ],
            skipped=False,
        )

        result = model_recommendation(
            profile={"has_labels": True, "n_documents": 25, "label_column": "label"},
            task_type="sentiment",
            evaluation_result=evaluation,
        )

        assert result.measured_skip_reason is None
        assert len(result.measured_on_user_data) == 2
        accuracies = {m.model_name: m.accuracy for m in result.measured_on_user_data}
        assert accuracies["distilbert-base-uncased-finetuned-sst-2-english"] == 0.92
        assert accuracies["textattack/bert-base-uncased-SST-2"] == 0.6

    def test_high_measured_accuracy_sharpens_fine_tune_note(self, monkeypatch):
        _mock_llm(monkeypatch, _GOOD_LLM_RESPONSE)

        evaluation = EvaluateCandidatesOutput(
            per_model=[CandidateEvaluation(model_name="distilbert-base-uncased-finetuned-sst-2-english", accuracy=0.95, f1=0.94, n_examples=200)],
            skipped=False,
        )

        result = model_recommendation(
            profile={"has_labels": True, "n_documents": 5000},
            task_type="sentiment",
            evaluation_result=evaluation,
        )

        assert "unlikely to be necessary" in result.fine_tune_note
        assert "does not perform training or fine-tuning" in result.fine_tune_note


class TestResearchNoteVariants:
    def test_research_not_requested_when_never_attempted(self, monkeypatch):
        _mock_llm(monkeypatch, _GOOD_LLM_RESPONSE)
        result = model_recommendation(
            profile={"has_labels": False, "n_documents": 10}, task_type="sentiment", research_evidence=None
        )
        assert result.external_research == []
        assert "not requested" in result.research_note.lower()

    def test_research_attempted_but_unavailable_reads_differently_from_not_requested(self, monkeypatch):
        # docs/AGENT_WORKFLOWS.md §8: "attempted but unavailable" must never be silently indistinguishable
        # from "never requested" — this is the found:false-from-research_models scenario.
        _mock_llm(monkeypatch, _GOOD_LLM_RESPONSE)
        result = model_recommendation(
            profile={"has_labels": False, "n_documents": 10},
            task_type="sentiment",
            research_evidence=None,
            research_attempted=True,
        )
        assert result.external_research == []
        assert "attempted" in result.research_note.lower()
        assert "not requested" not in result.research_note.lower()

    def test_research_populated_no_note_needed(self, monkeypatch):
        _mock_llm(monkeypatch, _GOOD_LLM_RESPONSE)
        evidence = [ResearchEvidence(claim="X", source_title="Y", source_url="https://y.example", snippet="z")]
        result = model_recommendation(
            profile={"has_labels": False, "n_documents": 10}, task_type="sentiment", research_evidence=evidence
        )
        assert len(result.external_research) == 1
        assert result.research_note is None


class TestOutputSchemaSeparation:
    def test_all_three_sections_always_present_and_distinct_fields(self, monkeypatch):
        _mock_llm(monkeypatch, _GOOD_LLM_RESPONSE)
        result = model_recommendation(profile={"has_labels": False, "n_documents": 10}, task_type="sentiment")

        assert hasattr(result, "measured_on_user_data")
        assert hasattr(result, "external_research")
        assert hasattr(result, "system_judgment")
        assert isinstance(result.measured_on_user_data, list)
        assert isinstance(result.external_research, list)
        assert isinstance(result.system_judgment, str)


class TestLLMFailureFallback:
    def test_llm_error_degrades_to_rule_based_response(self, monkeypatch):
        from llm.client import LLMError

        def raise_error():
            raise LLMError("provider down")

        monkeypatch.setattr(model_recommendation_module, "GroqLLMClient", raise_error)

        result = model_recommendation(profile={"has_labels": False, "n_documents": 10}, task_type="sentiment")

        assert result.degraded is True
        assert result.recommendation == "distilbert-base-uncased-finetuned-sst-2-english"
        assert len(result.rationale) >= 1
        assert "does not perform training or fine-tuning" in result.fine_tune_note

    def test_malformed_llm_json_also_degrades(self, monkeypatch):
        from unittest.mock import MagicMock

        mock_client = MagicMock()
        mock_client.complete.return_value = "not json"
        monkeypatch.setattr(model_recommendation_module, "GroqLLMClient", lambda: mock_client)

        result = model_recommendation(profile={"has_labels": False, "n_documents": 10}, task_type="sentiment")

        assert result.degraded is True

    def test_unknown_task_type_raises(self, monkeypatch):
        _mock_llm(monkeypatch, _GOOD_LLM_RESPONSE)
        with pytest.raises(ValueError):
            model_recommendation(profile={"has_labels": False, "n_documents": 10}, task_type="not_a_real_task")
