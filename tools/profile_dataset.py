"""profile_dataset — mandatory near-first step; see docs/TOOLS_AND_MODELS.md #1.

Deterministic (pandas/heuristics) except for an optional, lightweight language-id pass via `langdetect`
(the doc's stated lighter alternative to the XLM-RoBERTa language-id model, chosen to stay inside the
< 1.5s latency target for up to 5k rows — see docs/TOOLS_AND_MODELS.md #1 "Expected latency").
"""

import statistics

from ingestion.corpus import Corpus
from ingestion.store import corpus_store
from tools.schemas import ProfileDatasetOutput
from tools.timing import timed_tool

_LABEL_COLUMN_CANDIDATES = {"label", "labels", "class", "category", "sentiment", "rating", "target", "y"}
_MAX_LABEL_CARDINALITY = 20
_LANGDETECT_SAMPLE_SIZE = 20


def _resolve_texts(corpus: Corpus, column_hint: str | None) -> tuple[list[str], str | None, str | None]:
    """Returns (texts, text_column, reason). reason is set only on a resolvable-but-invalid column_hint."""
    if column_hint is None:
        default_column = corpus.documents[0].metadata.get("text_column") if corpus.documents else None
        texts = [doc.text for doc in corpus.documents]
        return texts, default_column, None

    if column_hint == "text":
        return [doc.text for doc in corpus.documents], "text", None

    available = corpus.documents[0].metadata.keys() if corpus.documents else []
    if column_hint not in available:
        return [], None, f"column_hint '{column_hint}' was not found in this corpus's columns."

    texts = [str(doc.metadata.get(column_hint, "")) for doc in corpus.documents]
    return texts, column_hint, None


def _length_distribution(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {}
    sorted_lengths = sorted(lengths)
    return {
        "min": float(sorted_lengths[0]),
        "p25": (
            float(statistics.quantiles(sorted_lengths, n=4)[0])
            if len(sorted_lengths) >= 2
            else float(sorted_lengths[0])
        ),
        "median": float(statistics.median(sorted_lengths)),
        "p75": (
            float(statistics.quantiles(sorted_lengths, n=4)[2])
            if len(sorted_lengths) >= 2
            else float(sorted_lengths[-1])
        ),
        "max": float(sorted_lengths[-1]),
    }


def _detect_language(texts: list[str]) -> str | None:
    sample = [t for t in texts[:_LANGDETECT_SAMPLE_SIZE] if t.strip()]
    if not sample:
        return None
    try:
        from langdetect import LangDetectException, detect
    except ImportError:
        return None
    votes: dict[str, int] = {}
    for text in sample:
        try:
            lang = detect(text)
        except LangDetectException:
            continue
        votes[lang] = votes.get(lang, 0) + 1
    if not votes:
        return None
    return max(votes, key=votes.get)


def _detect_labels(corpus: Corpus) -> tuple[bool, str | None, dict[str, float] | None]:
    if not corpus.documents:
        return False, None, None
    metadata_keys = corpus.documents[0].metadata.keys()
    label_column = next(
        (key for key in metadata_keys if key.lower() in _LABEL_COLUMN_CANDIDATES),
        None,
    )
    if label_column is None:
        return False, None, None

    values = [doc.metadata.get(label_column) for doc in corpus.documents if doc.metadata.get(label_column) is not None]
    unique_values = set(values)
    # A real label column repeats values across rows; a column with (near-)one-unique-value-per-row
    # (a free-text field, an id) is not categorical even if it happens to be under the cardinality cap.
    looks_categorical = len(unique_values) < len(values)
    if not values or len(unique_values) > _MAX_LABEL_CARDINALITY or not looks_categorical:
        return False, None, None

    counts: dict[str, float] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    distribution = {key: count / len(values) for key, count in counts.items()}
    return True, label_column, distribution


@timed_tool
def profile_dataset(corpus_ref: str, column_hint: str | None = None) -> ProfileDatasetOutput:
    corpus = corpus_store.get(corpus_ref)

    texts, text_column, reason = _resolve_texts(corpus, column_hint)

    if reason is not None:
        return ProfileDatasetOutput(
            n_documents=corpus.document_count,
            text_column=None,
            avg_length=None,
            length_distribution={},
            detected_language=None,
            has_labels=False,
            source_format=corpus.source_type.value,
            reason=reason,
        )

    lengths = [len(t) for t in texts]
    avg_length = statistics.mean(lengths) if lengths else None
    detected_language = _detect_language(texts)
    has_labels, label_column, class_distribution = _detect_labels(corpus)

    return ProfileDatasetOutput(
        n_documents=corpus.document_count,
        text_column=text_column,
        avg_length=avg_length,
        length_distribution=_length_distribution(lengths),
        detected_language=detected_language,
        has_labels=has_labels,
        label_column=label_column,
        class_distribution=class_distribution,
        source_format=corpus.source_type.value,
    )
