"""model_recommendation — docs/TOOLS_AND_MODELS.md #9, docs/MODEL_RECOMMENDATION.md.

This module is built in two independent layers:

1. A deterministic, rule-based candidate shortlist + fine-tune advisory (§2-3, §9) — no LLM, no external
   dependency, cheap. This is the "backbone the LLM prose is required to stay consistent with" (§3).
2. `model_recommendation()` itself — the LLM-grounded synthesis step wiring the rule engine together with
   `evaluate_candidates` and `research_models` output into the A/B/C schema-enforced structure (§7). Added
   once (1) and evaluate_candidates are solid.
"""

import json
import re
from typing import Any

from llm.client import GroqLLMClient, LLMError
from tools.schemas import (
    CandidateEvaluation,
    CandidateModel,
    EvaluateCandidatesOutput,
    ModelRecommendationOutput,
    ResearchEvidence,
    UserConstraints,
)
from tools.timing import timed_tool

# Per-task shortlist: (model_name, is_default). The default entry in every task is already the
# smaller/distilled model per docs/TOOLS_AND_MODELS.md's own "Default Pretrained Model Choices" table, so
# "favor distilled variants under latency/CPU constraints" (§3) naturally keeps it first; alternatives are
# the named larger/well-known models real users specifically ask about (e.g. "BERT vs DistilBERT").
#
# Sentiment's BERT-base alternative was verified live before being added here: textattack/bert-base-
# uncased-SST-2 outputs raw, unmapped "LABEL_0"/"LABEL_1" — evaluate_candidates' label normalization
# (tools/evaluate_candidates.py) would never match those against "positive"/"negative" ground truth,
# silently reporting ~0% accuracy for a model that might actually perform fine. Dropped rather than
# shipped broken; cardiffnlp's RoBERTa-base variant (verified clean "positive"/"negative" output) is kept
# as the sole named alternative, which still satisfies §3's "1-2 named alternatives."
_TASK_CANDIDATES: dict[str, list[tuple[str, bool]]] = {
    "sentiment": [
        ("distilbert-base-uncased-finetuned-sst-2-english", True),
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


# ---------------------------------------------------------------------------
# LLM-grounded synthesis — docs/MODEL_RECOMMENDATION.md §6-8
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json_object(raw: str) -> dict[str, Any]:
    fence_match = _JSON_FENCE_RE.search(raw)
    candidate = fence_match.group(1) if fence_match else raw.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise LLMError(f"response was not valid JSON: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise LLMError(f"response JSON was not an object: {parsed!r}")
    return parsed


def _fallback_confidence_note(profile: dict[str, Any], external_research: list[ResearchEvidence]) -> str:
    reasons = []
    if (profile.get("n_documents") or 0) < 50:
        reasons.append("small dataset size")
    if not external_research:
        reasons.append("no external research available")
    prefix = f"Lower confidence ({', '.join(reasons)}). " if reasons else ""
    return f"{prefix}This is a degraded, rule-based-only response — LLM synthesis was unavailable."


def _llm_synthesize(
    task_type: str,
    profile: dict[str, Any],
    user_constraints: UserConstraints | None,
    candidates: list[CandidateModel],
    measured: list[CandidateEvaluation],
    external_research: list[ResearchEvidence],
) -> tuple[str, list[str], str, str]:
    """Returns (recommendation, rationale, system_judgment, confidence_note). Raises LLMError on any
    failure (call/timeout/malformed response) so the caller's single fallback path handles all of them
    uniformly."""
    candidates_text = "\n".join(
        f"- {c.model_name} ({'default' if c.is_default else 'alternative'}): {c.reason}" for c in candidates
    )
    measured_text = (
        "\n".join(f"- {m.model_name}: accuracy={m.accuracy:.1%}, f1={m.f1:.1%}, n={m.n_examples}" for m in measured)
        if measured
        else "(none — no evaluation was run on the user's own data)"
    )
    research_text = (
        "\n".join(f"- {e.claim} — {e.source_title} ({e.source_url})" for e in external_research)
        if external_research
        else "(no external evidence found/requested)"
    )
    constraints_text = (
        f"latency_requirement={user_constraints.latency_requirement if user_constraints else None}, "
        f"compute_constraints={user_constraints.compute_constraints if user_constraints else None}"
    )

    prompt = f"""You are TextInsight's model recommendation writer for a {task_type} task.

Dataset profile: n_documents={profile.get('n_documents')}, has_labels={profile.get('has_labels')}, \
avg_length={profile.get('avg_length')}, detected_language={profile.get('detected_language')}.
User constraints: {constraints_text}

Rule-based candidate shortlist (deterministic — do not invent your own ranking or add candidates):
{candidates_text}

Measured on the user's own data (real numbers, if any):
{measured_text}

External research evidence (if any):
{research_text}

Rules you must follow:
- Never state a model is objectively "best" unless it was actually measured on the user's own data above.
- Phrase the recommendation as "Recommended based on dataset characteristics and stated constraints" (or \
equivalent) — never as an unqualified absolute claim.
- If you reference an external research claim, cite its source title inline.
- If measured results exist, prefer the best-measured candidate as your recommendation.
- Derive confidence_note from the concrete signals above (dataset size, whether evidence was found) — do \
not invent unrelated reasons.

Respond with ONLY a JSON object, no prose outside it, no markdown fences. ALL FOUR keys below are
required — never omit "system_judgment" or "confidence_note" even though they're prose, not data:
{{"recommendation": "<one model_name from the shortlist above, or the best-measured one>", \
"rationale": ["<reason 1>", "<reason 2>"], "system_judgment": "<REQUIRED: 2-4 sentence recommendation \
prose, do not skip this key>", "confidence_note": "<REQUIRED: one line on how confident this is and \
why, do not skip this key>"}}"""

    client = GroqLLMClient()
    raw = client.complete([{"role": "user", "content": prompt}])
    parsed = _parse_json_object(raw)

    recommendation = parsed.get("recommendation")
    rationale = parsed.get("rationale")
    system_judgment = parsed.get("system_judgment")
    confidence_note = parsed.get("confidence_note")

    if not (
        isinstance(recommendation, str)
        and recommendation
        and isinstance(rationale, list)
        and rationale
        and isinstance(system_judgment, str)
        and system_judgment
        and isinstance(confidence_note, str)
        and confidence_note
    ):
        raise LLMError(f"response JSON missing/malformed required fields: {parsed!r}")

    return recommendation, rationale, system_judgment, confidence_note


@timed_tool
def model_recommendation(
    profile: dict[str, Any],
    task_type: str,
    user_constraints: UserConstraints | None = None,
    research_evidence: list[ResearchEvidence] | None = None,
    research_attempted: bool = False,
    evaluation_result: EvaluateCandidatesOutput | None = None,
) -> ModelRecommendationOutput:
    candidates = generate_candidates(task_type, user_constraints)

    # Section A — docs/MODEL_RECOMMENDATION.md §7
    if evaluation_result is not None and not evaluation_result.skipped:
        measured = evaluation_result.per_model
        measured_skip_reason = None
        best = max(measured, key=lambda m: m.accuracy) if measured else None
        best_accuracy = best.accuracy if best else None
        default_recommendation = best.model_name if best else candidates[0].model_name
    else:
        measured = []
        best_accuracy = None
        default_recommendation = candidates[0].model_name
        if evaluation_result is not None:
            measured_skip_reason = evaluation_result.skip_reason
        else:
            measured_skip_reason = (
                "No evaluation was run — the dataset either has no labels or evaluate_candidates wasn't called."
            )

    # Section B — docs/AGENT_WORKFLOWS.md §8: "attempted but unavailable" must read differently from
    # "never requested" (research_evidence=None is ambiguous between them on its own).
    external_research = research_evidence or []
    if external_research:
        research_note = None
    elif research_attempted:
        research_note = "Research was attempted, but no external evidence was available for this query."
    else:
        research_note = "External research was not requested for this query."

    fine_tune_note = fine_tune_advisory_note(profile, best_measured_accuracy=best_accuracy)

    # Section C + top-level recommendation/rationale, LLM-grounded with a deterministic fallback
    try:
        recommendation, rationale, system_judgment, confidence_note = _llm_synthesize(
            task_type, profile, user_constraints, candidates, measured, external_research
        )
        degraded = False
    except LLMError:
        recommendation = default_recommendation
        rationale = [c.reason for c in candidates]
        system_judgment = (
            "Recommended based on dataset characteristics and stated constraints: "
            f"{default_recommendation}. (LLM synthesis was unavailable — this is the rule-based candidate "
            "list without prose rationale.)"
        )
        confidence_note = _fallback_confidence_note(profile, external_research)
        degraded = True

    return ModelRecommendationOutput(
        recommendation=recommendation,
        rationale=rationale,
        measured_on_user_data=measured,
        measured_skip_reason=measured_skip_reason,
        external_research=external_research,
        research_note=research_note,
        system_judgment=system_judgment,
        confidence_note=confidence_note,
        fine_tune_note=fine_tune_note,
        degraded=degraded,
    )
