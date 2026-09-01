import pytest

from ingestion.corpus import Document, SourceType
from ingestion.store import CorpusNotFoundError
from tests.helpers import register_corpus
from tools.profile_dataset import profile_dataset


class TestProfileDataset:
    def test_csv_corpus_profile(self):
        docs = [
            Document(id="0", text="Great product, fast shipping.", metadata={"text_column": "review_text", "rating": 5}),
            Document(id="1", text="Terrible support, broke fast.", metadata={"text_column": "review_text", "rating": 1}),
            Document(id="2", text="Average, nothing special here.", metadata={"text_column": "review_text", "rating": 3}),
        ]
        ref = register_corpus(docs)

        result = profile_dataset(corpus_ref=ref)

        assert result.n_documents == 3
        assert result.text_column == "review_text"
        assert result.avg_length > 0
        assert result.source_format == "csv"
        assert result.reason is None
        assert result.latency_ms >= 0
        assert set(result.length_distribution.keys()) == {"min", "p25", "median", "p75", "max"}

    def test_label_like_column_detected(self):
        docs = [
            Document(id=str(i), text=f"Review number {i} with some content.", metadata={"text_column": "text", "rating": i % 3})
            for i in range(6)
        ]
        ref = register_corpus(docs)

        result = profile_dataset(corpus_ref=ref)

        assert result.has_labels is True
        assert result.label_column == "rating"
        assert result.class_distribution is not None
        assert abs(sum(result.class_distribution.values()) - 1.0) < 1e-9

    def test_high_cardinality_column_not_treated_as_label(self):
        docs = [
            Document(id=str(i), text=f"Review {i}", metadata={"text_column": "text", "category": f"unique-{i}"})
            for i in range(10)
        ]
        ref = register_corpus(docs)

        result = profile_dataset(corpus_ref=ref)

        assert result.has_labels is False
        assert result.label_column is None

    def test_txt_corpus_has_no_text_column(self):
        docs = [Document(id="0", text="Just one document of prose here.")]
        ref = register_corpus(docs, source_type=SourceType.TXT, filename="test.txt")

        result = profile_dataset(corpus_ref=ref)

        assert result.text_column is None
        assert result.has_labels is False
        assert result.source_format == "txt"

    def test_invalid_column_hint_returns_reason_not_exception(self):
        docs = [Document(id="0", text="hello world", metadata={"text_column": "review_text"})]
        ref = register_corpus(docs)

        result = profile_dataset(corpus_ref=ref, column_hint="nonexistent_column")

        assert result.text_column is None
        assert result.reason is not None
        assert "nonexistent_column" in result.reason

    def test_column_hint_switches_text_source(self):
        docs = [
            Document(id="0", text="the default text column value", metadata={"text_column": "review_text", "title": "Short title"}),
        ]
        ref = register_corpus(docs)

        result = profile_dataset(corpus_ref=ref, column_hint="title")

        assert result.text_column == "title"
        assert result.avg_length == len("Short title")

    def test_unknown_corpus_ref_raises(self):
        with pytest.raises(CorpusNotFoundError):
            profile_dataset(corpus_ref="does-not-exist")
