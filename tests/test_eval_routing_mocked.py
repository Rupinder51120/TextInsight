"""Mocked/deterministic counterpart to tests/eval_routing.py, per docs/TESTING_STRATEGY.md §3's explicit
dual-mode guidance. This does NOT test whether the LLM routes correctly (it's mocked to always answer
"correctly") — it proves the harness's own JSON-plan-consumption, cache-filtering, and grading logic stays
correct in CI, independent of live Groq availability/determinism.
"""

import json
from unittest.mock import MagicMock

import agent.nodes as nodes
from tests.eval_routing import EVAL_CASES, run_eval


def test_mocked_deterministic_responses_all_pass(monkeypatch):
    responses = {case.query: json.dumps({"plan": case.required}) for case in EVAL_CASES}

    mock_client = MagicMock()
    mock_client.complete.side_effect = lambda messages: responses[messages[-1]["content"]]
    monkeypatch.setattr(nodes, "GroqLLMClient", lambda: mock_client)

    results = run_eval()

    failed = [r for r in results if not r.passed]
    assert not failed, f"harness/grading regression: {failed}"


def test_grading_rejects_wrong_tool_selection(monkeypatch):
    mock_client = MagicMock()
    mock_client.complete.return_value = json.dumps({"plan": ["summarize_text"]})  # wrong for a sentiment query
    monkeypatch.setattr(nodes, "GroqLLMClient", lambda: mock_client)

    from tests.eval_routing import EvalCase

    case = EvalCase(query="Analyze the sentiment", expected_intent="sentiment", required=["sentiment_analysis"])
    results = run_eval([case])

    assert results[0].passed is False
