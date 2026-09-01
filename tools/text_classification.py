"""text_classification — zero-shot classification, docs/TOOLS_AND_MODELS.md #3.

Model: valhalla/distilbart-mnli-12-3 (distilled MNLI, chosen for latency — zero-shot does one forward pass
per candidate label, the slowest default tool per docs/LATENCY_AND_PERFORMANCE.md).
"""

from ingestion.corpus import Document
from ingestion.store import corpus_store
from models.registry import get_zero_shot_pipeline
from tools.schemas import ClassificationDocumentResult, TextClassificationOutput
from tools.timing import timed_tool

_BATCH_SIZE = 8


def _select_documents(corpus_documents: list[Document], document_ids: list[str] | None) -> list[Document]:
    if document_ids is None:
        return corpus_documents
    wanted = set(document_ids)
    return [doc for doc in corpus_documents if doc.id in wanted]


@timed_tool
def text_classification(
    corpus_ref: str,
    candidate_labels: list[str],
    document_ids: list[str] | None = None,
) -> TextClassificationOutput:
    if not candidate_labels:
        raise ValueError("text_classification requires at least one candidate label.")

    corpus = corpus_store.get(corpus_ref)
    documents = _select_documents(corpus.documents, document_ids)
    documents = [doc for doc in documents if doc.text.strip()]

    if not documents:
        return TextClassificationOutput(per_document=[])

    pipe = get_zero_shot_pipeline()
    texts = [doc.text for doc in documents]
    raw_results = pipe(texts, candidate_labels=candidate_labels, batch_size=_BATCH_SIZE, truncation=True)
    if isinstance(raw_results, dict):  # pipeline returns a single dict, not a list, for one input text
        raw_results = [raw_results]

    per_document = [
        ClassificationDocumentResult(
            id=doc.id,
            label=raw["labels"][0],
            score=raw["scores"][0],
            all_scores=dict(zip(raw["labels"], raw["scores"])),
        )
        for doc, raw in zip(documents, raw_results)
    ]

    return TextClassificationOutput(per_document=per_document)
