"""model_recommendation — docs/TOOLS_AND_MODELS.md #9, docs/MODEL_RECOMMENDATION.md.

This module is built in two independent layers:

1. A deterministic, rule-based candidate shortlist + fine-tune advisory (§2-3, §9) — no LLM, no external
   dependency, cheap. This is the "backbone the LLM prose is required to stay consistent with" (§3).
2. `model_recommendation()` itself — the LLM-grounded synthesis step wiring the rule engine together with
   `evaluate_candidates` and `research_models` output into the A/B/C schema-enforced structure (§7). Added
   once (1) and evaluate_candidates are solid.
"""

from typing import Any

from tools.schemas import CandidateModel, UserConstraints

# Per-task shortlist: (model_name, is_default). The default entry in every task is already the
# smaller/distilled model per docs/TOOLS_AND_MODELS.md's own "Default Pretrained Model Choices" table, so
# "favor distilled variants under latency/CPU constraints" (§3) naturally keeps it first; alternatives are
# the named larger/well-known models real users specifically ask about (e.g. "BERT vs DistilBERT").
_TASK_CANDIDATES: dict[str, list[tuple[str, bool]]] = {
    "sentiment": [
        ("distilbert-base-uncased-finetuned-sst-2-english", True),
        ("textattack/bert-base-uncased-SST-2", False),
        ("cardiffnlp/twitter-roberta-base-sentiment-latest", False),
    ],
    "classification": [
        ("valhalla/distilbart-mnli-12-3", True),
        ("facebook/bart-large-mnli", False),
    ],
    "ner": [
        ("dslim/bert-base-NER", True),
        ("dslim/bert-large-NER", False),
    ],
    "summarization": [
        ("sshleifer/distilbart-cnn-12-6", True),
        ("facebook/bart-large-cnn", False),
    ],
    "embeddings": [
        ("sentence-transformers/all-MiniLM-L6-v2", True),
        ("sentence-transformers/all-mpnet-base-v2", False),
    ],
}

# "Large enough that fine-tuning is worth mentioning" — the docs don't pin an exact number (CLAUDE.md §6);
# chosen to match evaluate_candidates' own sample cap (docs/MODEL_RECOMMENDATION.md §6.5: "capped e.g.
# ≤500 labeled rows") so the two numbers stay consistent with each other.
_LARGE_DATASET_THRESHOLD = 500

_FINE_TUNE_DISCLAIMER = "This system does not perform training or fine-tuning; this is guidance only."


def generate_candidates(
    task_type: str,
    user_constraints: UserConstraints | None = None,
) -> list[CandidateModel]:
    """Deterministic, rule-based candidate shortlist — docs/MODEL_RECOMMENDATION.md §3. No LLM call."""
    if task_type not in _TASK_CANDIDATES:
        raise ValueError(
            f"no candidate shortlist defined for task_type={task_type!r}; "
            f"expected one of {sorted(_TASK_CANDIDATES)}"
        )

    constraints = user_constraints or UserConstraints()
    # Default assumption is CPU-only per docs/MODEL_RECOMMENDATION.md §2 ("matching the project's own
    # deployment target"); only an explicit gpu_available constraint relaxes the latency-sensitive framing.
    latency_sensitive = constraints.compute_constraints != "gpu_available" or constraints.latency_requirement in (
        "fast",
        "real-time",
    )

    candidates = []
    for model_name, is_default in _TASK_CANDIDATES[task_type]:
        if is_default and latency_sensitive:
            reason = "Default choice: smaller/distilled variant, fast on CPU — matches your latency/compute constraints."
        elif is_default:
            reason = "Default choice: strong accuracy/latency balance for this task."
        else:
            reason = "Named alternative some users specifically ask about; larger and typically slower than the default."
        candidates.append(CandidateModel(model_name=model_name, is_default=is_default, reason=reason))

    return candidates


def fine_tune_advisory_note(profile: dict[str, Any], best_measured_accuracy: float | None = None) -> str:
    """Rule-governed fine-tune-vs-pretrained framing — docs/MODEL_RECOMMENDATION.md §9. The disclaimer
    sentence is templated and always appended, regardless of which branch fires (never left to LLM
    recall)."""
    has_labels = bool(profile.get("has_labels"))
    n_documents = profile.get("n_documents") or 0

    if best_measured_accuracy is not None:
        if best_measured_accuracy >= 0.85:
            framing = (
                f"The best pretrained candidate already measured {best_measured_accuracy:.0%} accuracy on "
                "your own data — fine-tuning is unlikely to be necessary."
            )
        else:
            framing = (
                f"The best pretrained candidate measured {best_measured_accuracy:.0%} accuracy on your own "
                "data — if that isn't sufficient, fine-tuning could be worth exploring."
            )
    elif not has_labels or n_documents < _LARGE_DATASET_THRESHOLD:
        framing = (
            "Given the dataset is small and/or unlabeled, a pretrained/zero-shot approach is favored over "
            "fine-tuning."
        )
    else:
        framing = (
            "Given the dataset is reasonably large and labeled, fine-tuning could be worth exploring if "
            "pretrained accuracy isn't sufficient."
        )

    return f"{framing} {_FINE_TUNE_DISCLAIMER}"
