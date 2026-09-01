"""Real LangGraph node implementations — replaces Day 1's agent/graph.py stubs.

Design choices worth flagging (docs didn't pin these precisely — CLAUDE.md §6):

- Tool-calling flow uses docs/ARCHITECTURE.md §3.4's option (b): the LLM emits a structured JSON plan
  object (not native Groq tool-calling), which this module maps to real function calls. Chosen over (a)
  for Day 3 because it's simpler to test deterministically with mocked LLM responses (a hard requirement
  for tests/eval_routing.py's reproducible mode, docs/TESTING_STRATEGY.md §3).
- `understand_intent` and `plan_steps` stay two distinct nodes (per docs/ARCHITECTURE.md §0's explicit
  reasoning: "planning and synthesis are separately inspectable/testable steps"), but the *LLM call* itself
  is made once, in `understand_intent`, per docs/LATENCY_AND_PERFORMANCE.md §6 ("combined into a single LLM
  call wherever the plan is simple"). `plan_steps` then does purely deterministic refinement — profile-cache
  filtering, and (see below) plan repair after a replan signal — never a second LLM call.
- Any tool-raised ValueError (missing candidate_labels, filter_documents' criteria referencing a tool that
  hasn't run, an unavailable tool name) is treated as a *recoverable planning problem*, not a hard failure:
  `execute_tool` encodes it as `state["error"] = "REPLAN:<tool_name>:<reason>"`, `route_next` sends that to
  `plan_steps`, which repairs the plan (drops the unresolved step) and hands back to `execute_tool` — this
  is the concrete instance of docs/ARCHITECTURE.md §3.3's "route_next --> plan_steps: intermediate result
  requires re-planning (bounded)" edge. Any other exception is a hard failure routed to `handle_error`. This
  reuses AgentState.error's existing slot (via a parseable prefix) rather than adding a new state field, to
  keep AgentState an exact match to docs/ARCHITECTURE.md §3.1.
- The max-iteration guard (docs/ARCHITECTURE.md §3.3, `config.settings.max_tool_iterations`) counts
  successful tool executions (`step_index` increments). Because plan repair only ever *removes* the
  unresolved step (monotonic shrink, never growth), replanning cannot itself cause a runaway loop — the
  guard is a documented safety net, not the only thing preventing infinite loops.
- AgentState has no dedicated "truncated" flag; when the iteration guard cuts a plan short, `synthesize`
  prepends a plain-text notice to `final_answer` instead of adding a new state field.
"""

import json
import re
from typing import Any, Literal

from config import settings
from agent.state import AgentState
from agent.timing import timed
from ingestion.store import corpus_store
from llm.client import GroqLLMClient, LLMError
from tools import (
    filter_documents,
    generate_embeddings,
    profile_dataset,
    semantic_search,
    sentiment_analysis,
    summarize_text,
    text_classification,
)

# Full documented catalog (docs/TOOLS_AND_MODELS.md #1-10b), including tools not yet wired into
# _TOOL_FUNCTIONS below (named_entity_recognition, model_recommendation, evaluate_candidates,
# research_models — SHOULD-HAVE/Day-4 items). The LLM is told about the *complete* capability set so
# routing/planning can be evaluated against it now; execute_tool below only actually knows how to run the
# subset that exists as of Day 3, and gracefully replans around the rest (see module docstring).
_TOOL_CATALOG_PROMPT = """You are the planning component of TextInsight, an NLP analysis agent. Given a \
user's natural-language question about their uploaded dataset, decide which tool(s) to run, in order, to \
answer it.

Available tools:
- profile_dataset: dataset/text profiling (row count, text length, detected language, label presence). \
Cheap; run it first unless a profile is already cached for this corpus.
- sentiment_analysis: binary positive/negative sentiment per document.
- text_classification: zero-shot classification of documents into categories the user specifies (or that \
are implied by the query, e.g. "billing/technical/delivery/refund").
- named_entity_recognition: extracts organizations, people, locations, misc entities.
- summarize_text: summarizes one document, or a digest of many.
- generate_embeddings: builds a semantic search index for the corpus. Always pair it right before \
semantic_search.
- semantic_search: finds documents semantically similar to a natural-language query string.
- filter_documents: deterministic filtering of documents using a prior tool's output (e.g. keep only \
negative-sentiment documents, or the top semantic_search matches). Use it to chain analysis tools together.
- model_recommendation: recommends a pretrained model/approach for the dataset, given profiling and \
optionally research evidence.
- evaluate_candidates: measures real accuracy of candidate pretrained models against the user's own \
labeled data (only useful if the dataset has labels).
- research_models: searches the web for published benchmarks/model cards to support a model \
recommendation.

Keep the plan minimal: include only the tools actually needed to answer the question asked. Do not add \
sentiment_analysis or filter_documents just because the query mentions a word like "complaints" or \
"negative" — only add them if the query explicitly asks about sentiment, or explicitly asks to narrow down \
by sentiment before doing something else. A query that just asks to find/locate/search for documents about \
a topic is retrieval, not diagnosis — it needs semantic search alone, nothing more.

Common chains:
- Retrieval only ("find/search for/locate documents about X" — no "why", no request to analyze sentiment \
first): generate_embeddings, semantic_search. Nothing else. Example: "Find complaints about delayed \
delivery" -> ["generate_embeddings", "semantic_search"] (the word "complaints" here just names the topic to \
search for; it is not a request to run sentiment analysis).
- A diagnostic "why" question asking to explain/understand negative feedback (not just retrieve examples of \
it): profile_dataset, sentiment_analysis, filter_documents, generate_embeddings, semantic_search, \
summarize_text.
- "Show me only X and summarize/analyze them" (X is an explicit sentiment/category filter the user named): \
the relevant analysis tool, filter_documents, then the follow-up tool.
- A model-choice question: profile_dataset, evaluate_candidates (if labels likely exist), research_models, \
model_recommendation.

Respond with ONLY a JSON object of the form {"plan": ["tool_name", ...]}, using only tool names from the \
list above, in the order they should run. No prose, no markdown fences. If the query is empty, unclear, or \
cannot be answered with any of these tools, respond {"plan": []}."""

_ALL_TOOL_NAMES = {
    "profile_dataset",
    "sentiment_analysis",
    "text_classification",
    "named_entity_recognition",
    "summarize_text",
    "generate_embeddings",
    "semantic_search",
    "filter_documents",
    "model_recommendation",
    "evaluate_candidates",
    "research_models",
}

_TOOL_FUNCTIONS = {
    "profile_dataset": profile_dataset,
    "sentiment_analysis": sentiment_analysis,
    "text_classification": text_classification,
    "summarize_text": summarize_text,
    "generate_embeddings": generate_embeddings,
    "semantic_search": semantic_search,
    "filter_documents": filter_documents,
}

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_LABEL_SPLIT_RE = re.compile(r"[,/]| or ")


def _parse_plan(raw: str) -> list[str]:
    fence_match = _JSON_FENCE_RE.search(raw)
    candidate = fence_match.group(1) if fence_match else raw.strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {raw!r}") from exc

    if not isinstance(parsed, dict) or "plan" not in parsed:
        raise ValueError(f"LLM response JSON missing a 'plan' key: {parsed!r}")

    plan = parsed["plan"]
    if not isinstance(plan, list) or not all(isinstance(t, str) for t in plan):
        raise ValueError(f"'plan' must be a list of tool-name strings, got: {plan!r}")

    unknown = [t for t in plan if t not in _ALL_TOOL_NAMES]
    if unknown:
        raise ValueError(f"plan references unknown tool name(s): {unknown}")

    return plan


def _extract_candidate_labels(query: str) -> list[str] | None:
    """Heuristic: pull a slash/comma/'or'-separated label list following 'into' or 'labels:'."""
    match = re.search(r"(?:into|labels?:)\s+(.+?)(?:[.?!]|$)", query, re.IGNORECASE)
    if not match:
        return None
    tokens = [t.strip() for t in _LABEL_SPLIT_RE.split(match.group(1)) if t.strip()]
    return tokens or None


def _build_kwargs(tool_name: str, state: AgentState) -> dict[str, Any]:
    corpus_ref = state["corpus_ref"]
    query = state["user_query"]
    profile = state.get("profile")
    results = state["tool_results"]

    if tool_name == "profile_dataset":
        return {"corpus_ref": corpus_ref}

    if tool_name == "sentiment_analysis":
        kwargs: dict[str, Any] = {"corpus_ref": corpus_ref}
        if "filter_documents" in results:
            kwargs["document_ids"] = results["filter_documents"].document_ids
        if profile and profile.get("text_column"):
            kwargs["text_column"] = profile["text_column"]
        return kwargs

    if tool_name == "text_classification":
        labels = _extract_candidate_labels(query)
        if labels is None and profile and profile.get("has_labels") and profile.get("class_distribution"):
            labels = list(profile["class_distribution"].keys())
        if labels is None:
            raise ValueError("no candidate labels could be determined from the query or dataset")
        kwargs = {"corpus_ref": corpus_ref, "candidate_labels": labels}
        if "filter_documents" in results:
            kwargs["document_ids"] = results["filter_documents"].document_ids
        return kwargs

    if tool_name == "summarize_text":
        kwargs = {"corpus_ref": corpus_ref}
        if "filter_documents" in results:
            kwargs["document_ids"] = results["filter_documents"].document_ids
            kwargs["mode"] = "batch_digest"
        else:
            corpus = corpus_store.get(corpus_ref)
            kwargs["mode"] = "single" if corpus.document_count == 1 else "batch_digest"
        return kwargs

    if tool_name == "generate_embeddings":
        return {"corpus_ref": corpus_ref}

    if tool_name == "semantic_search":
        return {"corpus_ref": corpus_ref, "query": query, "top_k": 5}

    if tool_name == "filter_documents":
        if "sentiment_analysis" in results:
            return {
                "corpus_ref": corpus_ref,
                "criteria": {"from_tool": "sentiment_analysis", "field": "label", "equals": "negative"},
                "source_result": results["sentiment_analysis"],
            }
        if "semantic_search" in results:
            return {
                "corpus_ref": corpus_ref,
                "criteria": {"from_tool": "semantic_search", "top_k": 20},
                "source_result": results["semantic_search"],
            }
        raise ValueError("filter_documents has no prior sentiment_analysis or semantic_search result yet")

    raise ValueError(f"'{tool_name}' is not available yet")


@timed("understand_intent")
def understand_intent(state: AgentState) -> AgentState:
    query = state["user_query"]
    if not query or not query.strip():
        state["error"] = "Your message was empty — what would you like to know about your data?"
        return state

    try:
        client = GroqLLMClient()
        raw = client.complete(
            [
                {"role": "system", "content": _TOOL_CATALOG_PROMPT},
                {"role": "user", "content": query},
            ]
        )
        state["plan"] = _parse_plan(raw)
    except LLMError as exc:
        state["error"] = f"I couldn't process that request right now ({exc}). Please try again."
    except ValueError as exc:
        state["error"] = f"I couldn't understand how to plan for that request ({exc}). Could you rephrase it?"

    return state


@timed("plan_steps")
def plan_steps(state: AgentState) -> AgentState:
    error = state.get("error")
    if error and error.startswith("REPLAN:"):
        _, tool_name, _reason = error.split(":", 2)
        state["plan"] = [t for t in state["plan"] if t != tool_name]
        state["error"] = None
        return state

    if state.get("profile") is not None and "profile_dataset" in state["plan"]:
        state["plan"] = [t for t in state["plan"] if t != "profile_dataset"]
    state["step_index"] = 0
    return state


@timed("execute_tool")
def execute_tool(state: AgentState) -> AgentState:
    plan = state["plan"]
    step_index = state["step_index"]
    if step_index >= len(plan):
        return state

    tool_name = plan[step_index]
    tool_fn = _TOOL_FUNCTIONS.get(tool_name)

    try:
        if tool_fn is None:
            raise ValueError(f"'{tool_name}' is not available yet")
        kwargs = _build_kwargs(tool_name, state)
        result = tool_fn(**kwargs)
    except ValueError as exc:
        state["error"] = f"REPLAN:{tool_name}:{exc}"
        return state
    except Exception as exc:  # noqa: BLE001 — genuine tool failure, not a planning problem
        state["error"] = f"'{tool_name}' failed: {exc}"
        return state

    state["tool_results"][tool_name] = result
    if tool_name == "profile_dataset":
        state["profile"] = result.model_dump()
    state["step_index"] += 1
    state["latency"][f"tool:{tool_name}"] = getattr(result, "latency_ms", 0.0)
    return state


def route_next(state: AgentState) -> Literal["execute_tool", "plan_steps", "synthesize", "handle_error"]:
    """Conditional edge after execute_tool — see module docstring for the replan/guard rationale."""
    error = state.get("error")
    if error:
        return "plan_steps" if error.startswith("REPLAN:") else "handle_error"
    if state["step_index"] >= settings.max_tool_iterations:
        return "synthesize"
    if state["step_index"] >= len(state["plan"]):
        return "synthesize"
    return "execute_tool"


def route_after_understand(state: AgentState) -> Literal["plan_steps", "handle_error"]:
    return "handle_error" if state.get("error") else "plan_steps"


def _format_tool_results(state: AgentState) -> str:
    lines = []
    for tool_name, result in state["tool_results"].items():
        lines.append(f"- {tool_name}: {result.model_dump()}")
    return "\n".join(lines) if lines else "(no tool results)"


def _fallback_summary(state: AgentState) -> str:
    if not state["tool_results"]:
        return "I wasn't able to run any analysis for that request."
    parts = [f"Ran: {', '.join(state['tool_results'].keys())}."]
    for tool_name, result in state["tool_results"].items():
        parts.append(f"{tool_name} result: {result.model_dump()}")
    return " ".join(parts)


@timed("synthesize")
def synthesize(state: AgentState) -> AgentState:
    truncated = state["step_index"] >= settings.max_tool_iterations and state["step_index"] < len(state["plan"])

    try:
        client = GroqLLMClient()
        prompt = (
            "You are TextInsight's explanation step. Using ONLY the structured tool results below (never "
            "invent numbers or facts not present here), write a clear, concise natural-language answer to "
            "the user's question. If evidence came from research, label it as external. If nothing ran, "
            "say so honestly.\n\n"
            f"User question: {state['user_query']}\n\n"
            f"Tool results:\n{_format_tool_results(state)}"
        )
        answer = client.complete([{"role": "user", "content": prompt}])
    except LLMError:
        answer = _fallback_summary(state)

    if truncated:
        answer = (
            f"(Note: stopped after {settings.max_tool_iterations} analysis steps, the maximum for one "
            "turn — the answer below reflects what completed.)\n\n" + answer
        )

    state["final_answer"] = answer
    return state


@timed("handle_error")
def handle_error(state: AgentState) -> AgentState:
    state["final_answer"] = (
        f"I ran into a problem while working on that: {state['error']}. Could you rephrase or try again?"
    )
    return state
