class IngestionError(ValueError):
    """Raised for any unsupported, oversized, malformed, or empty upload.

    FastAPI (Day 5) is expected to catch this and return a structured 4xx — see
    docs/ARCHITECTURE.md §11 ("Ingestion errors ... structured 4xx from FastAPI ... agent is never invoked").
    """
