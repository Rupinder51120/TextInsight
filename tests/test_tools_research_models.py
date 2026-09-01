"""research_models — docs/TESTING_STRATEGY.md §8. Mocked ResearchClient (injected, not monkeypatched) for
the well-formed/empty/failure cases, plus a real degraded-mode check since this environment genuinely has
no TAVILY_API_KEY configured — that IS the live "API key missing" path, not a simulation of it.
"""

import sys
from unittest.mock import MagicMock

from research.client import Evidence, ResearchError
from tools.research_models import _build_query, research_models

# See tests/test_tools_sentiment_logic.py's comment: sys.modules is the only lookup immune to a future
# tools/__init__.py re-export shadow (module name == function name), so monkeypatch targets go through it.
research_models_module = sys.modules["tools.research_models"]


class FakeResearchClient:
    def __init__(self, results=None, raises=False):
        self._results = results or []
        self._raises = raises

    def search(self, query, max_results=5):
        if self._raises:
            raise ResearchError("simulated failure")
        return self._results


class TestBuildQuery:
    def test_topic_takes_precedence(self):
        assert _build_query("sentiment", ["a", "b"], "custom topic") == "custom topic"

    def test_falls_back_to_task_and_models(self):
        query = _build_query("sentiment", ["DistilBERT", "BERT"], None)
        assert "sentiment" in query
        assert "DistilBERT vs BERT" in query


class TestResearchModels:
    def test_well_formed_results_propagate_source_title_and_url(self, monkeypatch):
        fake_client = FakeResearchClient(
            results=[Evidence(title="BERT paper", url="https://arxiv.org/abs/bert", snippet="BERT is a transformer model.")]
        )
        monkeypatch.setattr(research_models_module, "GroqLLMClient", lambda: MagicMock(complete=lambda m: "BERT is a transformer-based model."))

        result = research_models(task_type="sentiment", candidate_models=["BERT"], client=fake_client)

        assert result.found is True
        assert len(result.evidence) == 1
        assert result.evidence[0].source_title == "BERT paper"
        assert result.evidence[0].source_url == "https://arxiv.org/abs/bert"
        assert result.evidence[0].claim  # non-empty

    def test_search_failure_returns_found_false_not_an_exception(self):
        fake_client = FakeResearchClient(raises=True)

        result = research_models(task_type="sentiment", client=fake_client)

        assert result.found is False
        assert result.evidence == []

    def test_empty_results_returns_found_false(self):
        fake_client = FakeResearchClient(results=[])

        result = research_models(task_type="sentiment", client=fake_client)

        assert result.found is False
        assert result.evidence == []

    def test_llm_condensation_failure_falls_back_to_snippet_not_a_crash(self, monkeypatch):
        from llm.client import LLMError

        fake_client = FakeResearchClient(
            results=[Evidence(title="X", url="https://x.example", snippet="A snippet about the model.")]
        )

        def raise_llm_error():
            raise LLMError("down")

        monkeypatch.setattr(research_models_module, "GroqLLMClient", raise_llm_error)

        result = research_models(task_type="sentiment", client=fake_client)

        assert result.found is True
        assert result.evidence[0].claim == "A snippet about the model."

    def test_real_environment_has_no_tavily_key_and_degrades_cleanly(self):
        # No `client=` injected here — exercises the real default TavilyResearchClient() construction path.
        # This environment genuinely has no TAVILY_API_KEY set, so this is the live degraded-mode behavior,
        # not a simulation of it.
        result = research_models(task_type="sentiment", candidate_models=["DistilBERT"])

        assert result.found is False
        assert result.evidence == []
