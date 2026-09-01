"""sentiment_analysis — docs/TOOLS_AND_MODELS.md #2.

Model: distilbert-base-uncased-finetuned-sst-2-english (binary POSITIVE/NEGATIVE — no neutral class; this
limitation is surfaced in the UI, per the doc, not hidden here).
"""

from ingestion.corpus import Document
from ingestion.store import corpus_store
from models.registry import get_sentiment_pipeline
from tools.schemas import SentimentAnalysisOutput, SentimentDocumentResult
from tools.timing import timed_tool

_MIN_TEXT_CHARS = 3
_BATCH_SIZE = 16


def _select_documents(corpus_documents: list[Document], document_ids: list[str] | None) -> list[Document]:
    if document_ids is None:
        return corpus_documents
    wanted = set(document_ids)
    return [doc for doc in corpus_documents if doc.id in wanted]


def _resolve_text(doc: Document, text_column: str | None) -> str:
    # `text_column` may be profile_dataset's *default* text_column (docs/TOOLS_AND_MODELS.md #2: "from
    # profile") — which is just the name of the column ingestion already used to build doc.text. Ingestion
    # deliberately excludes that column from doc.metadata (it's already doc.text), so a plain metadata
    # lookup for it would always come up empty. Only treat text_column as a genuine *alternate*-column
    # override when it names something other than the document's own primary column.
    if text_column is None or text_column == "text" or text_column == doc.metadata.get("text_column"):
        return doc.text
    return str(doc.metadata.get(text_column, ""))


@timed_tool
def sentiment_analysis(
    corpus_ref: str,
    document_ids: list[str] | None = None,
    text_column: str | None = None,
) -> SentimentAnalysisOutput:
    corpus = corpus_store.get(corpus_ref)
    documents = _select_documents(corpus.documents, document_ids)

    usable_ids: list[str] = []
    usable_texts: list[str] = []
    skipped_count = 0
    for doc in documents:
        text = _resolve_text(doc, text_column)
        if len(text.strip()) < _MIN_TEXT_CHARS:
            skipped_count += 1
            continue
        usable_ids.append(doc.id)
        usable_texts.append(text)

    if not usable_texts:
        return SentimentAnalysisOutput(per_document=[], distribution={}, skipped_count=skipped_count)

    pipe = get_sentiment_pipeline()
    raw_results = pipe(usable_texts, truncation=True, batch_size=_BATCH_SIZE)

    per_document = [
        SentimentDocumentResult(id=doc_id, label=raw["label"].lower(), score=raw["score"])
        for doc_id, raw in zip(usable_ids, raw_results)
    ]

    counts: dict[str, int] = {"positive": 0, "negative": 0}
    for result in per_document:
        counts[result.label] += 1
    total = len(per_document)
    distribution = {label: count / total for label, count in counts.items()}

    return SentimentAnalysisOutput(per_document=per_document, distribution=distribution, skipped_count=skipped_count)
