"""Unit tests for the real agent/nodes.py node implementations — mocked LLM/tools, no network or model
downloads. Live end-to-end coverage lives in tests/test_agent_graph_integration.py and tests/eval_routing.py.
"""

from unittest.mock import MagicMock

import agent.nodes as nodes
from agent.state import new_state
from llm.client import LLMError


def _state(query="Analyze the sentiment", **overrides):
    state = new_state(session_id="s", corpus_ref="c", user_query=query)
    state.update(overrides)
    return state


class TestUnderstandIntent:
    def test_empty_query_sets_error_without_calling_llm(self, monkeypatch):
        mock_client_cls = MagicMock()
        monkeypatch.setattr(nodes, "GroqLLMClient", mock_client_cls)

        result = nodes.understand_intent(_state(query="   "))

        assert result["error"] is not None
        mock_client_cls.assert_not_called()

    def test_valid_plan_response_sets_state_plan(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.complete.return_value = '{"plan": ["profile_dataset", "sentiment_analysis"]}'
        monkeypatch.setattr(nodes, "GroqLLMClient", lambda: mock_client)

        result = nodes.understand_intent(_state())

        assert result["plan"] == ["profile_dataset", "sentiment_analysis"]
        assert result["error"] is None

    def test_malformed_json_sets_error(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.complete.return_value = "not json at all"
        monkeypatch.setattr(nodes, "GroqLLMClient", lambda: mock_client)

        result = nodes.understand_intent(_state())

        assert result["error"] is not None

    def test_llm_error_sets_error(self, monkeypatch):
        def raise_error():
            raise LLMError("provider down")

        monkeypatch.setattr(nodes, "GroqLLMClient", raise_error)

        result = nodes.understand_intent(_state())

        assert result["error"] is not None
        assert "provider down" in result["error"]


class TestPlanSteps:
    def test_fresh_plan_with_no_cached_profile_is_unchanged(self):
        state = _state(plan=["profile_dataset", "sentiment_analysis"], profile=None)

        result = nodes.plan_steps(state)

        assert result["plan"] == ["profile_dataset", "sentiment_analysis"]
        assert result["step_index"] == 0

    def test_cached_profile_strips_profile_dataset_from_plan(self):
        state = _state(plan=["profile_dataset", "sentiment_analysis"], profile={"text_column": "review_text"})

        result = nodes.plan_steps(state)

        assert result["plan"] == ["sentiment_analysis"]

    def test_replan_signal_removes_offending_tool_and_clears_error(self):
        state = _state(
            plan=["text_classification", "summarize_text"],
            error="REPLAN:text_classification:no candidate labels",
        )

        result = nodes.plan_steps(state)

        assert result["plan"] == ["summarize_text"]
        assert result["error"] is None


class TestExecuteTool:
    def test_no_op_when_plan_exhausted(self):
        state = _state(plan=["profile_dataset"], step_index=1)

        result = nodes.execute_tool(state)

        assert result["step_index"] == 1
        assert result["tool_results"] == {}

    def test_successful_tool_call_updates_state(self, monkeypatch):
        fake_output = MagicMock()
        fake_output.latency_ms = 12.5
        fake_output.model_dump.return_value = {"n_documents": 3}
        mock_tool = MagicMock(return_value=fake_output)
        monkeypatch.setattr(nodes, "_TOOL_FUNCTIONS", {**nodes._TOOL_FUNCTIONS, "profile_dataset": mock_tool})

        state = _state(plan=["profile_dataset"], step_index=0)
        result = nodes.execute_tool(state)

        assert result["step_index"] == 1
        assert result["tool_results"]["profile_dataset"] is fake_output
        assert result["profile"] == {"n_documents": 3}
        assert result["latency"]["tool:profile_dataset"] == 12.5

    def test_value_error_sets_replan_signal(self, monkeypatch):
        mock_tool = MagicMock(side_effect=ValueError("no candidate labels"))
        monkeypatch.setattr(nodes, "_TOOL_FUNCTIONS", {**nodes._TOOL_FUNCTIONS, "sentiment_analysis": mock_tool})

        state = _state(plan=["sentiment_analysis"], step_index=0)
        result = nodes.execute_tool(state)

        assert result["error"] == "REPLAN:sentiment_analysis:no candidate labels"
        assert result["step_index"] == 0
        assert "sentiment_analysis" not in result["tool_results"]

    def test_build_kwargs_failure_also_sets_replan_signal(self):
        # _build_kwargs itself can raise ValueError (e.g. text_classification with no resolvable labels)
        # before the tool function is even called — same REPLAN handling applies.
        state = _state(query="Analyze the sentiment", plan=["text_classification"], step_index=0)

        result = nodes.execute_tool(state)

        assert result["error"].startswith("REPLAN:text_classification:")
        assert result["step_index"] == 0

    def test_unexpected_exception_sets_hard_error(self, monkeypatch):
        mock_tool = MagicMock(side_effect=RuntimeError("model crashed"))
        monkeypatch.setattr(nodes, "_TOOL_FUNCTIONS", {**nodes._TOOL_FUNCTIONS, "sentiment_analysis": mock_tool})

        state = _state(plan=["sentiment_analysis"], step_index=0)
        result = nodes.execute_tool(state)

        assert result["error"] == "'sentiment_analysis' failed: model crashed"
        assert not result["error"].startswith("'sentiment_analysis' failed: REPLAN")

    def test_unavailable_tool_name_triggers_replan(self):
        # named_entity_recognition is the one catalog tool deliberately not wired into execute_tool
        # (SHOULD-HAVE, deprioritized per PROJECT_SPEC.md §11) — a genuine "not available" case.
        state = _state(plan=["named_entity_recognition"], step_index=0)

        result = nodes.execute_tool(state)

        assert result["error"] == "REPLAN:named_entity_recognition:'named_entity_recognition' is not available yet"


class TestRouteNext:
    def test_replan_error_routes_to_plan_steps(self):
        state = _state(error="REPLAN:foo:bar")
        assert nodes.route_next(state) == "plan_steps"

    def test_hard_error_routes_to_handle_error(self):
        state = _state(error="'foo' failed: boom")
        assert nodes.route_next(state) == "handle_error"

    def test_plan_complete_routes_to_synthesize(self):
        state = _state(plan=["profile_dataset"], step_index=1)
        assert nodes.route_next(state) == "synthesize"

    def test_more_steps_remain_routes_to_execute_tool(self):
        state = _state(plan=["profile_dataset", "sentiment_analysis"], step_index=1)
        assert nodes.route_next(state) == "execute_tool"

    def test_max_iterations_forces_synthesize_even_with_plan_remaining(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "max_tool_iterations", 3)
        state = _state(plan=["a", "b", "c", "d", "e"], step_index=3)

        assert nodes.route_next(state) == "synthesize"


class TestRouteAfterUnderstand:
    def test_error_routes_to_handle_error(self):
        assert nodes.route_after_understand(_state(error="boom")) == "handle_error"

    def test_no_error_routes_to_plan_steps(self):
        assert nodes.route_after_understand(_state(error=None)) == "plan_steps"


class TestSynthesize:
    def test_llm_success_sets_final_answer(self, monkeypatch):
        mock_client = MagicMock()
        mock_client.complete.return_value = "Here is the analysis."
        monkeypatch.setattr(nodes, "GroqLLMClient", lambda: mock_client)

        state = _state(plan=["profile_dataset"], step_index=1, tool_results={})
        result = nodes.synthesize(state)

        assert result["final_answer"] == "Here is the analysis."

    def test_llm_failure_falls_back_to_templated_summary(self, monkeypatch):
        def raise_error():
            raise LLMError("down")

        monkeypatch.setattr(nodes, "GroqLLMClient", raise_error)

        state = _state(plan=[], step_index=0, tool_results={})
        result = nodes.synthesize(state)

        assert result["final_answer"] is not None
        assert "wasn't able to run any analysis" in result["final_answer"]

    def test_truncation_notice_prepended_when_iteration_guard_hit(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "max_tool_iterations", 2)
        mock_client = MagicMock()
        mock_client.complete.return_value = "Partial answer."
        monkeypatch.setattr(nodes, "GroqLLMClient", lambda: mock_client)

        state = _state(plan=["a", "b", "c", "d"], step_index=2, tool_results={})
        result = nodes.synthesize(state)

        assert "stopped after 2 analysis steps" in result["final_answer"]
        assert "Partial answer." in result["final_answer"]

    def test_no_truncation_notice_when_plan_actually_completed(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "max_tool_iterations", 6)
        mock_client = MagicMock()
        mock_client.complete.return_value = "Full answer."
        monkeypatch.setattr(nodes, "GroqLLMClient", lambda: mock_client)

        state = _state(plan=["a", "b"], step_index=2, tool_results={})
        result = nodes.synthesize(state)

        assert result["final_answer"] == "Full answer."


class TestHandleError:
    def test_produces_user_safe_message_referencing_error(self):
        state = _state(error="'sentiment_analysis' failed: model crashed")
        result = nodes.handle_error(state)

        assert result["final_answer"] is not None
        assert "model crashed" in result["final_answer"]
