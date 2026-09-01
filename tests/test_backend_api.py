"""FastAPI endpoint tests via TestClient — docs/TESTING_STRATEGY.md §2. Covers success and
validation-error paths for /upload, /query, /session/{id}/history.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client():
    return TestClient(app)


def _upload_reviews(client) -> dict:
    content = (FIXTURES / "csv" / "reviews.csv").read_bytes()
    resp = client.post("/upload", files={"file": ("reviews.csv", content, "text/csv")})
    assert resp.status_code == 200
    return resp.json()


class TestUploadEndpoint:
    def test_valid_csv_returns_profile_and_session(self, client):
        data = _upload_reviews(client)

        assert data["session_id"]
        assert data["corpus_ref"]
        assert data["document_count"] == 5
        assert data["source_format"] == "csv"
        assert data["profile"]["text_column"] == "review_text"

    def test_reusing_session_id_replaces_corpus(self, client):
        first = _upload_reviews(client)
        session_id = first["session_id"]

        txt_content = (FIXTURES / "txt" / "single_doc.txt").read_bytes()
        resp = client.post(
            "/upload",
            files={"file": ("single_doc.txt", txt_content, "text/plain")},
            data={"session_id": session_id},
        )
        assert resp.status_code == 200
        second = resp.json()

        assert second["session_id"] == session_id
        assert second["corpus_ref"] != first["corpus_ref"]
        assert second["source_format"] == "txt"

    def test_unsupported_extension_returns_400(self, client):
        resp = client.post("/upload", files={"file": ("bad.exe", b"not a real file", "application/octet-stream")})
        assert resp.status_code == 400
        assert "unsupported" in resp.json()["detail"].lower()

    def test_empty_file_returns_400(self, client):
        resp = client.post("/upload", files={"file": ("empty.csv", b"", "text/csv")})
        assert resp.status_code == 400

    def test_agent_is_never_invoked_on_ingestion_failure(self, client, monkeypatch):
        # docs/ARCHITECTURE.md §11: ingestion errors are structured 4xx, agent never invoked.
        from unittest.mock import MagicMock

        import backend.main as backend_main

        mock_run_agent = MagicMock()
        monkeypatch.setattr(backend_main, "run_agent", mock_run_agent)

        client.post("/upload", files={"file": ("empty.csv", b"", "text/csv")})

        mock_run_agent.assert_not_called()


class TestQueryEndpoint:
    def test_valid_query_returns_grounded_response(self, client, monkeypatch):
        from unittest.mock import MagicMock

        import backend.main as backend_main

        fake_result = {
            "plan": ["sentiment_analysis"],
            "tool_results": {},
            "final_answer": "All positive.",
            "error": None,
            "latency": {"understand_intent": 10.0},
            "profile": None,
        }
        monkeypatch.setattr(backend_main, "run_agent", MagicMock(return_value=fake_result))

        data = _upload_reviews(client)
        resp = client.post("/query", json={"session_id": data["session_id"], "query": "Analyze the sentiment"})

        assert resp.status_code == 200
        body = resp.json()
        assert body["final_answer"] == "All positive."
        assert body["plan"] == ["sentiment_analysis"]
        assert body["error"] is None

    def test_unknown_session_returns_404(self, client):
        resp = client.post("/query", json={"session_id": "does-not-exist", "query": "hi"})
        assert resp.status_code == 404

    def test_tool_results_are_json_serialized(self, client, monkeypatch):
        from unittest.mock import MagicMock

        import backend.main as backend_main
        from tools.schemas import SentimentAnalysisOutput, SentimentDocumentResult

        fake_output = SentimentAnalysisOutput(
            per_document=[SentimentDocumentResult(id="0", label="positive", score=0.9)],
            distribution={"positive": 1.0, "negative": 0.0},
        )
        fake_result = {
            "plan": ["sentiment_analysis"],
            "tool_results": {"sentiment_analysis": fake_output},
            "final_answer": "Positive.",
            "error": None,
            "latency": {},
            "profile": None,
        }
        monkeypatch.setattr(backend_main, "run_agent", MagicMock(return_value=fake_result))

        data = _upload_reviews(client)
        resp = client.post("/query", json={"session_id": data["session_id"], "query": "Analyze the sentiment"})

        body = resp.json()
        assert body["tool_results"]["sentiment_analysis"]["distribution"] == {"positive": 1.0, "negative": 0.0}


class TestHistoryEndpoint:
    def test_history_reflects_completed_turns(self, client, monkeypatch):
        from unittest.mock import MagicMock

        import backend.main as backend_main

        fake_result = {
            "plan": [],
            "tool_results": {},
            "final_answer": "An answer.",
            "error": None,
            "latency": {},
            "profile": None,
        }
        monkeypatch.setattr(backend_main, "run_agent", MagicMock(return_value=fake_result))

        data = _upload_reviews(client)
        client.post("/query", json={"session_id": data["session_id"], "query": "A question"})

        resp = client.get(f"/session/{data['session_id']}/history")
        assert resp.status_code == 200
        history = resp.json()["chat_history"]
        assert history == [
            {"role": "user", "content": "A question"},
            {"role": "assistant", "content": "An answer."},
        ]

    def test_unknown_session_returns_404(self, client):
        resp = client.get("/session/does-not-exist/history")
        assert resp.status_code == 404
