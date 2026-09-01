"""Shared upload validation — size limit and content sniffing, applied before any parsing.

Per docs/SECURITY_AND_RELIABILITY.md §1: file type is restricted by content sniffing (not just the
extension), and the size limit is enforced before any parsing begins.
"""

from config import settings
from ingestion.corpus import SourceType
from ingestion.errors import IngestionError

_PDF_MAGIC = b"%PDF-"


def validate_size(content: bytes, filename: str) -> None:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise IngestionError(
            f"{filename}: file is {len(content) / (1024 * 1024):.1f} MB, "
            f"exceeds the {settings.max_upload_mb:.0f} MB limit."
        )
    if len(content) == 0:
        raise IngestionError(f"{filename}: file is empty.")


def sniff_and_validate(content: bytes, filename: str, expected_type: SourceType) -> None:
    """Confirm file content actually matches expected_type, not just its extension."""
    validate_size(content, filename)

    if expected_type is SourceType.PDF:
        if not content.startswith(_PDF_MAGIC):
            raise IngestionError(f"{filename}: does not look like a valid PDF (missing %PDF header).")
        return

    # CSV/TXT: must be decodable text content, not binary data wearing a text extension.
    if b"\x00" in content:
        raise IngestionError(f"{filename}: contains binary data, not valid text/CSV content.")
    try:
        content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise IngestionError(f"{filename}: not valid UTF-8 text.") from exc
