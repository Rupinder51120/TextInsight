"""AgentState — the single state object threaded through every LangGraph node.

Definition matches docs/ARCHITECTURE.md §3.1 exactly. This is the only place tool/query data lives during a
turn — no node or tool holds its own module-level copy (see CLAUDE.md §4, "no global mutable state").
"""

from typing import Any, TypedDict


class AgentState(TypedDict):
    session_id: str
    corpus_ref: str
    chat_history: list[dict]
    user_query: str
    profile: dict | None
    plan: list[str]
    step_index: int
    tool_results: dict[str, Any]
    research_evidence: list[dict] | None
    latency: dict[str, float]
    final_answer: str | None
    error: str | None


def new_state(
    session_id: str,
    corpus_ref: str,
    user_query: str,
    chat_history: list[dict] | None = None,
    profile: dict | None = None,
) -> AgentState:
    """Construct a fresh AgentState for one turn. The only place default field values are decided.

    `profile` seeds a cached profile_dataset result from a prior turn in the same session (per
    docs/DATA_FLOW.md §3) — plan_steps (agent/nodes.py) already skips re-running profile_dataset whenever
    state["profile"] is populated, so passing it in here is what makes that cache-reuse actually span
    separate backend requests, not just steps within one graph run.
    """
    return AgentState(
        session_id=session_id,
        corpus_ref=corpus_ref,
        chat_history=chat_history or [],
        user_query=user_query,
        profile=profile,
        plan=[],
        step_index=0,
        tool_results={},
        research_evidence=None,
        latency={},
        final_answer=None,
        error=None,
    )
