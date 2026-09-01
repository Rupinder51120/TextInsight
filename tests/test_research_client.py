"""ResearchClient / TavilyResearchClient — mocked HTTP-equivalent (mocked tavily.TavilyClient), per
docs/TESTING_STRATEGY.md §1 ("LLMClient and ResearchClient adapters tested with mocked HTTP responses")."""

from unittest.mock import MagicMock

import pytest

from research.client import ResearchError, TavilyResearchClient


class TestTavilyResearchClient:
    def test_missing_api_key_raises_research_error(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "tavily_api_key", "")
        with pytest.raises(ResearchError):
            TavilyResearchClient()

    def test_search_returns_evidence_from_results(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "tavily_api_key", "fake-key")
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {
            "results": [
                {"title": "DistilBERT model card", "url": "https://hf.co/distilbert", "content": "A distilled version of BERT."},
            ]
        }
        monkeypatch.setattr("tavily.TavilyClient", lambda **kwargs: mock_tavily)

        client = TavilyResearchClient()
        results = client.search("distilbert benchmark", max_results=3)

        assert len(results) == 1
        assert results[0].title == "DistilBERT model card"
        assert results[0].url == "https://hf.co/distilbert"
        assert "distilled version" in results[0].snippet
        mock_tavily.search.assert_called_once_with(query="distilbert benchmark", max_results=3)

    def test_search_failure_raises_research_error(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "tavily_api_key", "fake-key")
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = Exception("network down")
        monkeypatch.setattr("tavily.TavilyClient", lambda **kwargs: mock_tavily)

        client = TavilyResearchClient()
        with pytest.raises(ResearchError):
            client.search("anything")

    def test_empty_results_returns_empty_list(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "tavily_api_key", "fake-key")
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {"results": []}
        monkeypatch.setattr("tavily.TavilyClient", lambda **kwargs: mock_tavily)

        client = TavilyResearchClient()
        assert client.search("anything") == []
