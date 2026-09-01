"""Full LangGraph conditional graph — docs/ARCHITECTURE.md §3.3. Replaces Day 1's stub skeleton
(`understand_intent_stub` / single unconditional `synthesize` call) with the real six-node graph.
"""

from langgraph.graph import END, START, StateGraph

from agent.nodes import (
    execute_tool,
    handle_error,
    plan_steps,
    route_after_understand,
    route_next,
    synthesize,
    understand_intent,
)
from agent.state import AgentState, new_state


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("understand_intent", understand_intent)
    graph.add_node("plan_steps", plan_steps)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("synthesize", synthesize)
    graph.add_node("handle_error", handle_error)

    graph.add_edge(START, "understand_intent")
    graph.add_conditional_edges(
        "understand_intent",
        route_after_understand,
        {"plan_steps": "plan_steps", "handle_error": "handle_error"},
    )
    graph.add_edge("plan_steps", "execute_tool")
    graph.add_conditional_edges(
        "execute_tool",
        route_next,
        {
            "execute_tool": "execute_tool",
            "plan_steps": "plan_steps",
            "synthesize": "synthesize",
            "handle_error": "handle_error",
        },
    )
    graph.add_edge("synthesize", END)
    graph.add_edge("handle_error", END)
    return graph.compile()


def run_agent(
    session_id: str,
    corpus_ref: str,
    user_query: str,
    chat_history: list[dict] | None = None,
    profile: dict | None = None,
) -> AgentState:
    app = build_graph()
    state = new_state(
        session_id=session_id,
        corpus_ref=corpus_ref,
        user_query=user_query,
        chat_history=chat_history,
        profile=profile,
    )
    return app.invoke(state)
