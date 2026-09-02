"""TXT -> Corpus.

Per docs/DATA_FLOW.md §1: "TXT becomes one document (or line-split for multi-record text files,
config-driven)". This loader auto-detects which mode applies by checking whether non-blank lines
individually read as complete, standalone records: each ends in sentence-terminal punctuation (. ! ? or a
closing quote/paren after one). Multi-record files (one review/ticket per line) satisfy this on nearly every
line; ordinary hard-wrapped prose (an essay/article) does not, since only the last line of each paragraph
ends a sentence — the rest wrap mid-sentence. A pure line-length or line-count heuristic can't tell these
apart (wrapped prose lines are just as short as one-per-line records), which is why punctuation-ending is
used instead. Documented here as the concrete rule since the docs left the exact threshold unspecified
(CLAUDE.md §6).
"""

from config import settings
from ingestion.corpus import Corpus, Document, SourceType, make_corpus_ref
from ingestion.errors import IngestionError
from ingestion.validation import sniff_and_validate

_SENTENCE_END_CHARS = (".", "!", "?", '"', "'", ")")
_RECORD_LINE_FRACTION_THRESHOLD = 0.8


def load_txt(content: bytes, filename: str) -> Corpus:
    sniff_and_validate(content, filename, SourceType.TXT)

    text = content.decode("utf-8")
    lines = [line.strip() for line in text.splitlines()]
    non_blank_lines = [line for line in lines if line]

    if not non_blank_lines:
        raise IngestionError(f"{filename}: file contains no text.")

    record_like = sum(1 for line in non_blank_lines if line.endswith(_SENTENCE_END_CHARS))
    record_fraction = record_like / len(non_blank_lines)
    multi_record = len(non_blank_lines) > 1 and record_fraction >= _RECORD_LINE_FRACTION_THRESHOLD

    if multi_record:
        records = non_blank_lines
    else:
        records = [text.strip()]

    original_count = len(records)
    truncated = original_count > settings.max_rows
    if truncated:
        records = records[: settings.max_rows]

    documents = [Document(id=str(i), text=record, metadata={"line_index": i}) for i, record in enumerate(records)]

    return Corpus(
        corpus_ref=make_corpus_ref(),
        source_type=SourceType.TXT,
        source_filename=filename,
        documents=documents,
        truncated=truncated,
        original_document_count=original_count if truncated else None,
    )
