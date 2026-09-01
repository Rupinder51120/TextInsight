"""Max-iteration guard — docs/ARCHITECTURE.md §3.3: "A hard max-iteration guard (e.g., 6 tool executions
per turn) prevents runaway loops; if exceeded, the agent synthesizes with whatever results exist and flags
the truncation." Required Day 3 test per docs/FIVE_DAY_BUILD_PLAN.md.

Runs the real compiled graph (agent/graph.py) with mocked tools/LLM: a plan longer than
config.settings.max_tool_iterations, where every tool call succeeds, to prove the guard — not a tool
failure — is what stops execution.
"""

from unittest.mock import MagicMock

import agent.nodes as nodes
from agent.graph import build_graph
from agent.state import new_state


def _fake_tool_result(n):
    output = MagicMock()
    output.latency_ms = 1.0
    output.model_dump.return_value = {"n": n}
    return output


def test_plan_longer_than_guard_stops_at_the_limit_not_at_plan_end(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "max_tool_iterations", 3)

    long_plan = ["profile_dataset"] * 8  # same tool repeated; only count matters here
    monkeypatch.setattr(
        nodes,
        "_TOOL_FUNCTIONS",
        {**nodes._TOOL_FUNCTIONS, "profile_dataset": MagicMock(side_effect=[_fake_tool_result(i) for i in range(8)])},
    )

    mock_llm = MagicMock()
    mock_llm.complete.return_value = '{"plan": ' + str(long_plan).replace("'", '"') + "}"
    monkeypatch.setattr(nodes, "GroqLLMClient", lambda: mock_llm)

    app = build_graph()
    state = new_state(session_id="s", corpus_ref="c", user_query="do a lot of steps")
    result = app.invoke(state)

    assert result["step_index"] == 3
    assert result["error"] is None
    assert "stopped after 3 analysis steps" in result["final_answer"]


def test_plan_shorter_than_guard_completes_normally(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "max_tool_iterations", 6)

    monkeypatch.setattr(
        nodes,
        "_TOOL_FUNCTIONS",
        {**nodes._TOOL_FUNCTIONS, "profile_dataset": MagicMock(return_value=_fake_tool_result(0))},
    )

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = ['{"plan": ["profile_dataset"]}', "All done."]
    monkeypatch.setattr(nodes, "GroqLLMClient", lambda: mock_llm)

    app = build_graph()
    state = new_state(session_id="s", corpus_ref="c", user_query="one simple step")
    result = app.invoke(state)

    assert result["step_index"] == 1
    assert "stopped after" not in result["final_answer"]
