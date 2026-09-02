"""evaluate_candidates skip-behavior and metric-computation logic — mocked pipeline, no download.
Correctness against a real labeled fixture with real models is in
tests/test_tools_evaluate_candidates_real.py, per docs/TESTING_STRATEGY.md §7.
"""

import sys
from unittest.mock import MagicMock

from ingestion.corpus import Document
from tests.helpers import register_corpus
from tools.evaluate_candidates import _compute_metrics, _normalize_label, evaluate_candidates

# See tests/test_tools_sentiment_logic.py's comment: sys.modules is the only lookup immune to the
# tools/__init__.py re-export shadow, so monkeypatch targets must go through it, not a dotted string path.
evaluate_candidates_module = sys.modules["tools.evaluate_candidates"]


class TestSkipBehavior:
    def test_no_labels_skips_with_reason(self):
        docs = [Document(id=str(i), text=f"doc {i}") for i in range(30)]
        ref = register_corpus(docs)

        result = evaluate_candidates(corpus_ref=ref, profile={"has_labels": False}, candidate_models=["m1"])

        assert result.skipped is True
        assert result.per_model == []
        assert "no labels" in result.skip_reason.lower()

    def test_too_few_labels_skips_with_reason(self):
        docs = [
            Document(id=str(i), text=f"doc {i}", metadata={"label": "positive" if i % 2 else "negative"})
            for i in range(10)  # below the 20-example minimum
        ]
        ref = register_corpus(docs)

        result = evaluate_candidates(
            corpus_ref=ref,
            profile={"has_labels": True, "label_column": "label"},
            candidate_models=["m1"],
        )

        assert result.skipped is True
        assert "too few" in result.skip_reason.lower()
        assert "10" in result.skip_reason

    def test_never_fabricates_numbers_when_skipped(self):
        result = evaluate_candidates(
            corpus_ref=register_corpus([Document(id="0", text="x")]),
            profile={"has_labels": False},
            candidate_models=["m1", "m2"],
        )
        assert result.per_model == []


class TestMetricsComputation:
    def test_perfect_predictions_score_1_0(self):
        accuracy, f1 = _compute_metrics(["positive", "negative", "positive"], ["positive", "negative", "positive"])
        assert accuracy == 1.0
        assert f1 == 1.0

    def test_all_wrong_predictions_score_0(self):
        accuracy, f1 = _compute_metrics(["positive", "negative"], ["negative", "positive"])
        assert accuracy == 0.0
        assert f1 == 0.0

    def test_partial_accuracy_computed_correctly(self):
        accuracy, _ = _compute_metrics(
            ["positive", "positive", "negative", "negative"],
            ["positive", "negative", "negative", "negative"],
        )
        assert accuracy == 0.75


class TestLabelNormalization:
    def test_known_synonyms_normalized(self):
        assert _normalize_label("POSITIVE") == "positive"
        assert _normalize_label("Neg") == "negative"
        assert _normalize_label("neutral") == "neutral"

    def test_unknown_labels_left_as_lowercased_literal(self):
        assert _normalize_label("Billing") == "billing"
        assert _normalize_label(1) == "1"


class TestEvaluateCandidatesRunsPerModel:
    def test_calls_pipeline_once_per_candidate_and_reports_correct_n_examples(self, monkeypatch):
        mock_pipe = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.9}] * 25)
        monkeypatch.setattr(evaluate_candidates_module, "get_pipeline", lambda task, model: mock_pipe)

        docs = [Document(id=str(i), text=f"review {i} here", metadata={"label": "positive"}) for i in range(25)]
        ref = register_corpus(docs)

        result = evaluate_candidates(
            corpus_ref=ref,
            profile={"has_labels": True, "label_column": "label"},
            candidate_models=["model-a", "model-b"],
        )

        assert result.skipped is False
        assert len(result.per_model) == 2
        assert mock_pipe.call_count == 2
        assert all(m.n_examples == 25 for m in result.per_model)
        assert all(m.accuracy == 1.0 for m in result.per_model)  # all predicted positive, all labeled positive

    def test_sample_size_caps_examples_used(self, monkeypatch):
        mock_pipe = MagicMock(return_value=[{"label": "POSITIVE", "score": 0.9}] * 5)
        monkeypatch.setattr(evaluate_candidates_module, "get_pipeline", lambda task, model: mock_pipe)

        docs = [Document(id=str(i), text=f"review {i} here", metadata={"label": "positive"}) for i in range(30)]
        ref = register_corpus(docs)

        result = evaluate_candidates(
            corpus_ref=ref,
            profile={"has_labels": True, "label_column": "label"},
            candidate_models=["model-a"],
            sample_size=5,
        )

        assert result.per_model[0].n_examples == 5
