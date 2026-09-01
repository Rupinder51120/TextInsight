"""filter_documents — the deterministic chaining tool, docs/TOOLS_AND_MODELS.md #8. No LLM call, ever.

`source_result` is the actual prior tool output being filtered on (e.g. a SentimentAnalysisOutput or a
SemanticSearchOutput) — not part of the documented Input schema (FilterDocumentsInput has only corpus_ref
and criteria), but required by the function itself; agent/nodes.py's execute_tool supplies it from
AgentState.tool_results, since only the agent (not this tool) knows which prior result criteria.from_tool
refers to.

Raises ValueError when criteria references a tool result that isn't available — per the doc: "criteria
references a tool that hasn't run yet → clear error surfaced to route_next, which should trigger
re-planning rather than crash." agent/nodes.py's execute_tool interprets any ValueError from a tool this
way (see its module docstring).
"""

from typing import Any

from ingestion.store import corpus_store
from tools.schemas import FilterDocumentsOutput
from tools.timing import timed_tool


@timed_tool
def filter_documents(corpus_ref: str, criteria: dict[str, Any], source_result: Any) -> FilterDocumentsOutput:
    corpus_store.get(corpus_ref)  # validates corpus_ref exists

    if source_result is None:
        from_tool = criteria.get("from_tool", "<unspecified>")
        raise ValueError(f"filter_documents: no result yet from '{from_tool}' to filter on.")

    if hasattr(source_result, "per_document"):
        field = criteria.get("field", "label")
        equals = criteria.get("equals")
        document_ids = [item.id for item in source_result.per_document if getattr(item, field, None) == equals]
    elif hasattr(source_result, "results"):
        top_k = criteria.get("top_k")
        items = source_result.results[:top_k] if top_k else source_result.results
        document_ids = [item.id for item in items]
    else:
        raise ValueError(
            f"filter_documents: don't know how to filter on a result of type {type(source_result).__name__}."
        )

    return FilterDocumentsOutput(document_ids=document_ids, count=len(document_ids))
