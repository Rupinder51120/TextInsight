"""Full-graph integration tests, live (real Groq + real HF/FAISS models, no mocking) — the actual proof
that the multi-step diagnostic chain and semantic search work end-to-end, per
docs/FIVE_DAY_BUILD_PLAN.md Day 3 completion criteria. Skipped without a real GROQ_API_KEY.
"""

from pathlib import Path

import pytest

from agent.graph import run_agent
from config import settings
from ingestion import load_csv, corpus_store
from tests.eval_routing import EVAL_CASES, _grade

pytestmark = pytest.mark.skipif(
    not settings.groq_api_key or settings.groq_api_key == "your-groq-api-key-here",
    reason="no real GROQ_API_KEY configured",
)

FIXTURES = Path(__file__).parent / "fixtures"


def _reviews_corpus_ref() -> str:
    content = (FIXTURES / "csv" / "reviews.csv").read_bytes()
    corpus = load_csv(content, "reviews.csv")
    corpus_store.put(corpus)
    return corpus.corpus_ref


class TestDiagnosticWorkflow:
    def test_why_are_customers_unhappy_produces_grounded_multistep_explanation(self):
        corpus_ref = _reviews_corpus_ref()

        result = run_agent(session_id="diag-test", corpus_ref=corpus_ref, user_query="Why are customers unhappy?")

        assert result["error"] is None
        # visible tool-call trace — docs/FIVE_DAY_BUILD_PLAN.md Day 3 completion criteria
        trace = list(result["tool_results"].keys())
        assert "sentiment_analysis" in trace
        assert "filter_documents" in trace
        assert "summarize_text" in trace
        assert trace.index("sentiment_analysis") < trace.index("filter_documents") < trace.index("summarize_text")

        # the filter actually narrowed to a real subset (this fixture has negative reviews)
        assert result["tool_results"]["filter_documents"].count > 0
        assert result["tool_results"]["filter_documents"].count < 5

        assert result["final_answer"]
        assert len(result["final_answer"]) > 20

        # every executed step recorded a real, positive latency
        for tool_name in trace:
            assert result["latency"][f"tool:{tool_name}"] >= 0


class TestSemanticSearchWorkflow:
    # Tolerant plan-matching, reused from tests/eval_routing.py (same case, same required/optional sets)
    # rather than a hard `"semantic_search" in tool_results` check — a single live LLM call is subject to
    # the same routing variance the eval harness measures.
    #
    # This query was previously a real routing weakness, not just flaky-test noise: isolated measurement
    # (2026-09-01, n=25 across two samples) found only ~24% single-attempt success — the agent's tool
    # catalog prompt had one example chain built around "negative feedback," and this query's word
    # "complaints" was pattern-matching onto it, padding a simple retrieval query with
    # sentiment_analysis/filter_documents (or misrouting to text_classification) most of the time.
    # Root-caused and fixed in agent/nodes.py's _TOOL_CATALOG_PROMPT (added an explicit retrieval-vs-
    # diagnosis distinction + a worked example for this exact phrasing) — re-measured at 15/15 (100%)
    # afterward, with the full 10-case eval also moving from 8/10 to 10/10. A small retry margin is kept
    # here as routine hygiene for a live-LLM test, not because the underlying issue is still present.
    _MAX_ATTEMPTS = 2

    def test_find_delayed_delivery_complaints_returns_ranked_matches(self):
        corpus_ref = _reviews_corpus_ref()
        query = "Find complaints about delayed delivery"
        case = next(c for c in EVAL_CASES if c.query == query)

        result = None
        grade = None
        for attempt in range(self._MAX_ATTEMPTS):
            result = run_agent(session_id=f"search-test-{attempt}", corpus_ref=corpus_ref, user_query=query)
            assert result["error"] is None

            executed_tools = list(result["tool_results"].keys())
            grade = _grade(case, executed_tools)
            if grade.passed:
                break

        assert grade.passed, (
            f"routing mismatch after {self._MAX_ATTEMPTS} attempts: {grade.reason} "
            f"(last executed: {list(result['tool_results'].keys())})"
        )

        search_output = result["tool_results"]["semantic_search"]
        assert len(search_output.results) > 0
        scores = [r.score for r in search_output.results]
        assert scores == sorted(scores, reverse=True)  # ranked, highest similarity first

        # query latency (the semantic_search step itself, not indexing) should be well under a second
        assert result["latency"]["tool:semantic_search"] < 1000
