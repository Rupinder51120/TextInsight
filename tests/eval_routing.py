"""Agent routing evaluation — docs/TESTING_STRATEGY.md §3. Protected MUST-HAVE per
docs/FIVE_DAY_BUILD_PLAN.md: "this eval set is run automatically as part of the test suite (using
mocked/deterministic LLM responses for reproducibility, plus a manual/optional run against the live Groq
model to catch prompt drift)".

This module provides both:
- `EVAL_CASES` + `run_eval(...)`: the shared harness, runnable as a script (`python -m tests.eval_routing`)
  against the LIVE Groq model — this is what actually measures whether the agent routes correctly, and is
  what a "pass rate" report should be based on.
- `test_eval_routing_live` (in this file, pytest-collected): runs the harness live and prints/asserts a
  floor pass rate. Requires GROQ_API_KEY.
- tests/test_eval_routing_mocked.py: the reproducible, network-free counterpart the doc calls for — proves
  the harness's own plan-consumption/matching logic (not the LLM's routing judgement) stays correct in CI.

Only `understand_intent` + `plan_steps` run for each case (not the full graph) — the eval set tests
*tool-sequence planning*, per TESTING_STRATEGY §3's "exact-match tool-sequence accuracy" metric, not full
execution (several expected tools — model_recommendation, evaluate_candidates, research_models,
named_entity_recognition — aren't wired into execute_tool until later days; see agent/nodes.py's docstring).
"""

from dataclasses import dataclass, field

from agent.nodes import plan_steps, understand_intent
from agent.state import new_state


@dataclass
class EvalCase:
    query: str
    expected_intent: str
    required: list[str]  # must all appear, in this relative order
    optional: set[str] = field(default_factory=set)  # may or may not appear either way


# profile_dataset is near-universally cache-conditional ("*" in the doc's table) across every row, so it's
# folded into every case's optional set once here rather than repeated 10 times.
_ALWAYS_OPTIONAL = {"profile_dataset"}


EVAL_CASES: list[EvalCase] = [
    EvalCase(
        query="Analyze the sentiment",
        expected_intent="sentiment",
        required=["sentiment_analysis"],
        optional=set(),
    ),
    EvalCase(
        query="Classify these complaints into billing/technical/delivery/refund",
        expected_intent="classification",
        required=["text_classification"],
        optional=set(),
    ),
    EvalCase(
        query="Extract organizations and people",
        expected_intent="ner",
        required=["named_entity_recognition"],
        optional=set(),
    ),
    EvalCase(
        query="Summarize these documents",
        expected_intent="summarization",
        required=["summarize_text"],
        optional=set(),
    ),
    EvalCase(
        query="Find complaints about delayed delivery",
        expected_intent="semantic_search",
        required=["semantic_search"],
        optional={"generate_embeddings"},
    ),
    EvalCase(
        query="Why are customers unhappy?",
        expected_intent="diagnostic_explanation",
        required=["sentiment_analysis", "filter_documents", "summarize_text"],
        optional={"generate_embeddings", "semantic_search"},
    ),
    EvalCase(
        query="Should I use BERT or DistilBERT?",
        expected_intent="model_recommendation (research)",
        required=["model_recommendation"],
        optional={"evaluate_candidates", "research_models"},
    ),
    EvalCase(
        query="Should I use a pretrained model or fine-tune?",
        expected_intent="model_recommendation",
        required=["model_recommendation"],
        optional={"research_models"},
    ),
    EvalCase(
        query="Show me only negative reviews and summarize them",
        expected_intent="multi-step (explicit)",
        required=["sentiment_analysis", "filter_documents", "summarize_text"],
        optional=set(),
    ),
    EvalCase(
        query="Which model gives the best latency for this task?",
        expected_intent="model_recommendation (constraint-driven)",
        required=["research_models", "model_recommendation"],
        optional=set(),
    ),
]


@dataclass
class CaseResult:
    case: EvalCase
    actual_plan: list[str]
    passed: bool
    reason: str = ""


def _grade(case: EvalCase, actual_plan: list[str]) -> CaseResult:
    allowed = set(case.required) | case.optional | _ALWAYS_OPTIONAL
    unexpected = [t for t in actual_plan if t not in allowed]
    if unexpected:
        return CaseResult(case, actual_plan, False, f"unexpected tool(s) selected: {unexpected}")

    positions = []
    for tool in case.required:
        if tool not in actual_plan:
            return CaseResult(case, actual_plan, False, f"missing required tool: {tool}")
        positions.append(actual_plan.index(tool))
    if positions != sorted(positions):
        return CaseResult(case, actual_plan, False, f"required tools out of order: {case.required}")

    return CaseResult(case, actual_plan, True)


def run_eval(cases: list[EvalCase] = EVAL_CASES) -> list[CaseResult]:
    """Runs understand_intent + plan_steps for each case against a fresh (uncached-profile) state, using
    whatever GroqLLMClient is currently wired (live unless the caller has monkeypatched it)."""
    results = []
    for case in cases:
        state = new_state(session_id="eval", corpus_ref="eval-corpus-placeholder", user_query=case.query)
        state = understand_intent(state)
        if state.get("error"):
            results.append(CaseResult(case, [], False, f"understand_intent error: {state['error']}"))
            continue
        state = plan_steps(state)
        results.append(_grade(case, state["plan"]))
    return results


def format_report(results: list[CaseResult]) -> str:
    lines = []
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"[{status}] {r.case.query!r} -> {r.actual_plan}" + (f"  ({r.reason})" if r.reason else ""))
    pass_rate = sum(r.passed for r in results) / len(results) if results else 0.0
    lines.append(f"\nPass rate: {sum(r.passed for r in results)}/{len(results)} ({pass_rate:.0%})")
    return "\n".join(lines)


if __name__ == "__main__":
    results = run_eval()
    print(format_report(results))
