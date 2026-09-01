from pathlib import Path

import pytest

from ingestion import IngestionError, SourceType, load_csv, load_pdf, load_txt

FIXTURES = Path(__file__).parent / "fixtures"


def _read(*parts: str) -> bytes:
    return (FIXTURES.joinpath(*parts)).read_bytes()


class TestCsvLoader:
    def test_valid_csv_produces_one_document_per_row(self):
        corpus = load_csv(_read("csv", "reviews.csv"), "reviews.csv")
        assert corpus.source_type is SourceType.CSV
        assert corpus.document_count == 5
        assert corpus.documents[0].metadata["text_column"] == "review_text"
        assert "shipping" in corpus.documents[0].text.lower()

    def test_non_text_columns_carried_as_metadata(self):
        corpus = load_csv(_read("csv", "reviews.csv"), "reviews.csv")
        assert corpus.documents[0].metadata["product"] == "Widget A"
        assert corpus.documents[0].metadata["rating"] == 5

    def test_no_text_column_raises(self):
        with pytest.raises(IngestionError):
            load_csv(_read("csv", "no_text_column.csv"), "no_text_column.csv")

    def test_empty_csv_raises(self):
        with pytest.raises(IngestionError):
            load_csv(_read("csv", "empty.csv"), "empty.csv")

    def test_malformed_csv_raises(self):
        malformed = b'a,b,c\n1,2\n3,4,5,6\n'
        with pytest.raises(IngestionError):
            load_csv(malformed, "malformed.csv")

    def test_binary_content_rejected(self):
        with pytest.raises(IngestionError):
            load_csv(b"\x00\x01\x02not really a csv", "fake.csv")


class TestTxtLoader:
    def test_multi_record_file_splits_by_line(self):
        corpus = load_txt(_read("txt", "multi_record.txt"), "multi_record.txt")
        assert corpus.document_count == 5

    def test_wrapped_prose_stays_one_document(self):
        corpus = load_txt(_read("txt", "single_doc.txt"), "single_doc.txt")
        assert corpus.document_count == 1
        assert "TextInsight" in corpus.documents[0].text

    def test_empty_txt_raises(self):
        with pytest.raises(IngestionError):
            load_txt(_read("txt", "empty.txt"), "empty.txt")


class TestPdfLoader:
    def test_text_pdf_extracts_cleanly(self):
        corpus = load_pdf(_read("pdf", "sample.pdf"), "sample.pdf")
        assert corpus.document_count == 1
        assert "TextInsight" in corpus.documents[0].text

    def test_scanned_pdf_with_no_text_raises(self):
        with pytest.raises(IngestionError):
            load_pdf(_read("pdf", "blank_no_text.pdf"), "blank_no_text.pdf")

    def test_non_pdf_content_rejected(self):
        with pytest.raises(IngestionError):
            load_pdf(b"this is not a pdf", "fake.pdf")


class TestSizeLimit:
    def test_oversized_file_rejected_before_parsing(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "max_upload_mb", 0.000001)
        with pytest.raises(IngestionError):
            load_csv(_read("csv", "reviews.csv"), "reviews.csv")
