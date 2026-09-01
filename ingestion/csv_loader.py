"""CSV -> Corpus.

Text-column detection here is a cheap heuristic (longest-average-length object/string column) so ingestion
can produce Documents immediately. profile_dataset (Day 2, per docs/DATA_FLOW.md §1) is the authoritative
detector and reports/refines this when the choice is ambiguous — ingestion's job is only to get a reasonable
default, not the final word.
"""

import io

import numpy as np
import pandas as pd

from config import settings
from ingestion.corpus import Corpus, Document, SourceType, make_corpus_ref
from ingestion.errors import IngestionError
from ingestion.validation import sniff_and_validate


def _detect_text_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if df[c].dtype == object]
    if not candidates:
        raise IngestionError(
            "No text-like column found in this CSV (all columns are numeric/other) — "
            "cannot build a text corpus from it."
        )
    avg_lengths = {}
    for c in candidates:
        non_null = df[c].dropna().astype(str)
        avg_lengths[c] = non_null.str.len().mean() if not non_null.empty else 0.0
    return max(avg_lengths, key=avg_lengths.get)


def _to_jsonable(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_csv(content: bytes, filename: str) -> Corpus:
    sniff_and_validate(content, filename, SourceType.CSV)

    try:
        df = pd.read_csv(io.BytesIO(content))
    except pd.errors.EmptyDataError as exc:
        raise IngestionError(f"{filename}: CSV has no data.") from exc
    except pd.errors.ParserError as exc:
        raise IngestionError(f"{filename}: malformed CSV ({exc}).") from exc

    if df.empty or len(df.columns) == 0:
        raise IngestionError(f"{filename}: CSV has no rows.")

    text_column = _detect_text_column(df)

    original_count = len(df)
    truncated = original_count > settings.max_rows
    if truncated:
        df = df.head(settings.max_rows)

    documents = [
        Document(
            id=str(idx),
            text=str(row[text_column]) if pd.notna(row[text_column]) else "",
            metadata={
                "row_index": int(idx),
                "text_column": text_column,
                **{k: _to_jsonable(v) for k, v in row.items() if k != text_column},
            },
        )
        for idx, row in df.iterrows()
    ]

    non_empty = [d for d in documents if d.text.strip()]
    if not non_empty:
        raise IngestionError(f"{filename}: detected text column '{text_column}' has no non-empty values.")

    return Corpus(
        corpus_ref=make_corpus_ref(),
        source_type=SourceType.CSV,
        source_filename=filename,
        documents=documents,
        truncated=truncated,
        original_document_count=original_count if truncated else None,
    )
