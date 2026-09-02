"""PDF -> Corpus, via PyMuPDF (fitz) — see docs/TECH_STACK.md "PDF: PyMuPDF (fitz)".

Per docs/DATA_FLOW.md §1: "PDF pages extracted and either kept as one document or page-segmented, based on
length." This loader segments per-page once the combined extracted text exceeds _SINGLE_DOC_MAX_CHARS,
otherwise keeps the whole PDF as one document. Threshold is an explicit assumption (CLAUDE.md §6) since the
docs don't pin an exact number.

Scanned/image-only PDFs (no extractable text) are rejected with a clear error, never silently returning an
empty corpus — see docs/SECURITY_AND_RELIABILITY.md §1 and §3 ("explicitly *not* claiming OCR support").
"""

import fitz  # PyMuPDF

from config import settings
from ingestion.corpus import Corpus, Document, SourceType, make_corpus_ref
from ingestion.errors import IngestionError
from ingestion.validation import sniff_and_validate

_SINGLE_DOC_MAX_CHARS = 5000


def load_pdf(content: bytes, filename: str) -> Corpus:
    sniff_and_validate(content, filename, SourceType.PDF)

    try:
        pdf = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises its own RuntimeError/fitz.FileDataError on corrupt PDFs
        raise IngestionError(f"{filename}: could not open as a PDF ({exc}).") from exc

    try:
        original_page_count = pdf.page_count
        truncated = original_page_count > settings.max_pdf_pages
        page_limit = min(original_page_count, settings.max_pdf_pages)

        pages = [pdf[i].get_text().strip() for i in range(page_limit)]
    finally:
        pdf.close()

    non_empty_pages = [p for p in pages if p]
    if not non_empty_pages:
        raise IngestionError(
            f"{filename}: no extractable text found (likely a scanned/image-only PDF — OCR is not supported)."
        )

    combined_len = sum(len(p) for p in non_empty_pages)

    if combined_len <= _SINGLE_DOC_MAX_CHARS:
        documents = [Document(id="0", text="\n\n".join(non_empty_pages), metadata={"page_count": page_limit})]
    else:
        documents = [
            Document(id=str(i), text=page, metadata={"page_number": i}) for i, page in enumerate(pages) if page
        ]

    return Corpus(
        corpus_ref=make_corpus_ref(),
        source_type=SourceType.PDF,
        source_filename=filename,
        documents=documents,
        truncated=truncated,
        original_document_count=original_page_count if truncated else None,
    )
