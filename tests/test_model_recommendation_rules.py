"""Rule-based candidate shortlist + fine-tune advisory — docs/MODEL_RECOMMENDATION.md §3, §9. Fully
deterministic, no LLM/model involved."""

import pytest

from tools.model_recommendation import fine_tune_advisory_note, generate_candidates
from tools.schemas import UserConstraints


class TestGenerateCandidates:
    def test_unknown_task_type_raises(self):
        with pytest.raises(ValueError):
            generate_candidates("not_a_real_task")

    def test_sentiment_shortlist_has_a_single_default_first(self):
        candidates = generate_candidates("sentiment")

        assert candidates[0].model_name == "distilbert-base-uncased-finetuned-sst-2-english"
        assert candidates[0].is_default is True
        assert sum(c.is_default for c in candidates) == 1
        assert len(candidates) >= 2  # default + at least one named alternative

    def test_every_candidate_has_a_non_empty_reason(self):
        for task_type in ("sentiment", "classification", "ner", "summarization", "embeddings"):
            for candidate in generate_candidates(task_type):
                assert candidate.reason.strip() != ""

    def test_cpu_only_default_assumption_favors_distilled_reason(self):
        candidates = generate_candidates("sentiment", user_constraints=None)
        default = next(c for c in candidates if c.is_default)
        assert "distilled" in default.reason.lower() or "cpu" in default.reason.lower()

    def test_explicit_gpu_and_no_latency_requirement_relaxes_framing(self):
        constraints = UserConstraints(compute_constraints="gpu_available", latency_requirement=None)
        candidates = generate_candidates("sentiment", user_constraints=constraints)
        default = next(c for c in candidates if c.is_default)
        assert "matches your latency/compute constraints" not in default.reason


class TestFineTuneAdvisoryNote:
    def test_disclaimer_always_present_small_unlabeled(self):
        note = fine_tune_advisory_note({"has_labels": False, "n_documents": 10})
        assert "does not perform training or fine-tuning" in note
        assert "small and/or unlabeled" in note

    def test_disclaimer_always_present_large_labeled(self):
        note = fine_tune_advisory_note({"has_labels": True, "n_documents": 5000})
        assert "does not perform training or fine-tuning" in note
        assert "fine-tuning could be worth exploring" in note

    def test_high_measured_accuracy_sharpens_framing(self):
        note = fine_tune_advisory_note({"has_labels": True, "n_documents": 5000}, best_measured_accuracy=0.95)
        assert "does not perform training or fine-tuning" in note
        assert "unlikely to be necessary" in note
        assert "95%" in note

    def test_low_measured_accuracy_still_offers_fine_tune_as_worth_considering(self):
        note = fine_tune_advisory_note({"has_labels": True, "n_documents": 5000}, best_measured_accuracy=0.55)
        assert "does not perform training or fine-tuning" in note
        assert "worth exploring" in note
        assert "55%" in note
