from unittest.mock import MagicMock

import pytest

from llm.client import GroqLLMClient, LLMError


@pytest.fixture
def fake_groq_api_key(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "groq_api_key", "fake-key-for-tests")


def _make_client_with_mocked_chat(monkeypatch, mock_chat):
    monkeypatch.setattr("langchain_groq.ChatGroq", lambda **kwargs: mock_chat)


class TestGroqLLMClient:
    def test_missing_api_key_raises_llm_error(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "groq_api_key", "")
        with pytest.raises(LLMError):
            GroqLLMClient()

    def test_complete_returns_response_content(self, fake_groq_api_key, monkeypatch):
        mock_chat = MagicMock()
        mock_chat.invoke.return_value = MagicMock(content="hello from groq")
        _make_client_with_mocked_chat(monkeypatch, mock_chat)

        client = GroqLLMClient()
        result = client.complete([{"role": "user", "content": "hi"}])

        assert result == "hello from groq"
        mock_chat.invoke.assert_called_once()

    def test_non_retryable_error_raises_immediately(self, fake_groq_api_key, monkeypatch):
        mock_chat = MagicMock()
        auth_error = Exception("invalid api key")
        auth_error.status_code = 401
        mock_chat.invoke.side_effect = auth_error
        _make_client_with_mocked_chat(monkeypatch, mock_chat)

        client = GroqLLMClient()
        with pytest.raises(LLMError):
            client.complete([{"role": "user", "content": "hi"}])

        assert mock_chat.invoke.call_count == 1

    def test_retryable_error_is_retried_then_succeeds(self, fake_groq_api_key, monkeypatch):
        mock_chat = MagicMock()
        rate_limit_error = Exception("rate limited")
        rate_limit_error.status_code = 429
        mock_chat.invoke.side_effect = [rate_limit_error, MagicMock(content="ok on retry")]
        _make_client_with_mocked_chat(monkeypatch, mock_chat)
        monkeypatch.setattr(GroqLLMClient, "_BACKOFF_BASE_SECONDS", 0.0)

        client = GroqLLMClient()
        result = client.complete([{"role": "user", "content": "hi"}])

        assert result == "ok on retry"
        assert mock_chat.invoke.call_count == 2

    def test_retries_exhausted_raises_llm_error(self, fake_groq_api_key, monkeypatch):
        mock_chat = MagicMock()
        server_error = Exception("server error")
        server_error.status_code = 500
        mock_chat.invoke.side_effect = server_error
        _make_client_with_mocked_chat(monkeypatch, mock_chat)
        monkeypatch.setattr(GroqLLMClient, "_BACKOFF_BASE_SECONDS", 0.0)

        client = GroqLLMClient()
        with pytest.raises(LLMError):
            client.complete([{"role": "user", "content": "hi"}])

        assert mock_chat.invoke.call_count == GroqLLMClient._MAX_RETRIES
