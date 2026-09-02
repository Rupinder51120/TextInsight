"""evaluate_candidates — docs/TOOLS_AND_MODELS.md #10a, docs/MODEL_RECOMMENDATION.md §6.5.

The priority item for Day 4: runs each shortlisted *pretrained* candidate as inference (no parameter
updates, no training loop) against a sample of the user's own labeled data and scores it against ground
truth — this is what makes Section A of model_recommendation's output a real measured number instead of a
permanent placeholder.

Label normalization: only well-known sentiment word forms (positive/negative/neutral and pos/neg/neu
abbreviations) are canonicalized; anything else (including numeric/star-rating labels) is compared as a
literal lowercased string. Numeric labels are deliberately NOT auto-mapped to positive/negative — e.g. "1"
could mean a binary positive class or the worst end of a 1-5 star scale, and guessing wrong would silently
corrupt every accuracy number this tool produces. This is a scope limitation (CLAUDE.md §6), not an
oversight: it's honest about what it can score rather than guessing.

Metrics (accuracy, macro-F1) are computed by hand rather than via scikit-learn — CLAUDE.md's locked stack
doesn't include a classical-ML library, and the computation itself is a few lines of counting, not a
modeling approach in its own right.
"""

from ingestion.store import corpus_store
from models.registry import get_pipeline
from tools.schemas import CandidateEvaluation, EvaluateCandidatesOutput
from tools.timing import timed_tool

_MIN_LABELED_EXAMPLES = 20
_DEFAULT_SAMPLE_SIZE = 500
_BATCH_SIZE = 16

_LABEL_SYNONYMS = {
    "positive": "positive",
    "pos": "positive",
    "negative": "negative",
    "neg": "negative",
    "neutral": "neutral",
    "neu": "neutral",
}


def _normalize_label(raw) -> str:
    text = str(raw).strip().lower()
    return _LABEL_SYNONYMS.get(text, text)


def _compute_metrics(y_true: list[str], y_pred: list[str]) -> tuple[float, float]:
    n = len(y_true)
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / n if n else 0.0

    labels = set(y_true) | set(y_pred)
    f1_scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1_scores.append(f1)
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return accuracy, macro_f1


@timed_tool
def evaluate_candidates(
    corpus_ref: str,
    profile: dict,
    candidate_models: list[str],
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
) -> EvaluateCandidatesOutput:
    if not profile.get("has_labels"):
        return EvaluateCandidatesOutput(
            per_model=[],
            skipped=True,
            skip_reason="Your dataset has no labels, so no evaluation could be run.",
        )

    label_column = profile.get("label_column")
    corpus = corpus_store.get(corpus_ref)

    labeled = [
        (doc.text, doc.metadata.get(label_column))
        for doc in corpus.documents
        if doc.text.strip() and doc.metadata.get(label_column) is not None
    ]

    if len(labeled) < _MIN_LABELED_EXAMPLES:
        return EvaluateCandidatesOutput(
            per_model=[],
            skipped=True,
            skip_reason=(
                f"Only {len(labeled)} labeled examples were found — too few for a meaningful comparison "
                f"(minimum {_MIN_LABELED_EXAMPLES})."
            ),
        )

    sample = labeled[:sample_size]
    texts = [t for t, _ in sample]
    ground_truth = [_normalize_label(label) for _, label in sample]

    per_model = []
    for model_name in candidate_models:
        pipe = get_pipeline("sentiment-analysis", model_name)
        raw_results = pipe(texts, truncation=True, batch_size=_BATCH_SIZE)
        predictions = [_normalize_label(r["label"]) for r in raw_results]

        accuracy, f1 = _compute_metrics(ground_truth, predictions)
        per_model.append(CandidateEvaluation(model_name=model_name, accuracy=accuracy, f1=f1, n_examples=len(sample)))

    return EvaluateCandidatesOutput(per_model=per_model, skipped=False)
