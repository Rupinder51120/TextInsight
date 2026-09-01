"""Document/Corpus models — the normalized representation every ingestion loader produces.

No module-level mutable state here or anywhere in ingestion/: a Corpus is a plain, immutable-by-convention
Pydantic value. Callers (session state, Day 5) are responsible for storing it keyed by corpus_ref — this
module never holds onto data itself (see CLAUDE.md §4, "no global mutable state").
"""

import uuid
from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    CSV = "csv"
    TXT = "txt"
    PDF = "pdf"


class Document(BaseModel):
    id: str
    text: str
    metadata: dict = Field(default_factory=dict)


class Corpus(BaseModel):
    corpus_ref: str
    source_type: SourceType
    source_filename: str
    documents: list[Document]
    truncated: bool = False
    original_document_count: int | None = None

    @property
    def document_count(self) -> int:
        return len(self.documents)


def make_corpus_ref() -> str:
    """A fresh, unique pointer for a newly ingested corpus (see ARCHITECTURE.md §3.1)."""
    return uuid.uuid4().hex
