"""Full-app smoke test — docs/FIVE_DAY_BUILD_PLAN.md Day 5: "full-app smoke test covering each of the 10
example use cases." The 10 queries are docs/TESTING_STRATEGY.md §3's routing evaluation set (also used by
tests/eval_routing.py) — docs/PROJECT_SPEC.md §6 ("User Journeys") only has 5 items, not 10, despite being
the doc FIVE_DAY_BUILD_PLAN.md and README_PLAN.md both cite; TESTING_STRATEGY.md §3 is the only 10-item
list anywhere in the docs, so it's used here as the intended set (documented assumption, CLAUDE.md §6).

Live, real Groq + real HF/FAISS models, through the actual FastAPI app (TestClient) — not a direct
agent.graph call — so this exercises upload, session continuity, and JSON serialization together, not just
agent routing (already covered live by tests/eval_routing.py and tests/test_agent_graph_integration.py).
Skipped without a real GROQ_API_KEY.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from config import settings
from tests.eval_routing import EVAL_CASES

pytestmark = pytest.mark.skipif(
    not settings.groq_api_key or settings.groq_api_key == "your-groq-api-key-here",
    reason="no real GROQ_API_KEY configured",
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def client():
    return TestClient(app)


def _upload(client, filename: str) -> str:
    content = (FIXTURES / "csv" / filename).read_bytes()
    resp = client.post("/upload", files={"file": (filename, content, "text/csv")})
    assert resp.status_code == 200
    return resp.json()["session_id"]


@pytest.mark.parametrize("case", EVAL_CASES, ids=[c.query for c in EVAL_CASES])
def test_example_query_produces_a_grounded_response(client, case):
    # labeled_sentiment.csv has real labels, which matters for the model-recommendation queries'
    # evaluate_candidates step; the plain reviews.csv fixture is unlabeled.
    filename = "labeled_sentiment.csv" if "model" in case.expected_intent or "recommendation" in case.expected_intent else "reviews.csv"
    session_id = _upload(client, filename)

    resp = client.post("/query", json={"session_id": session_id, "query": case.query})

    assert resp.status_code == 200
    body = resp.json()
    # error=None is required, not just a non-empty final_answer — handle_error's templated fallback
    # message is ALSO non-empty, so checking final_answer alone can't tell a real success apart from a
    # gracefully-degraded LLM failure (e.g. a rate limit). named_entity_recognition (not wired into
    # execute_tool) still resolves to error=None: plan_steps' REPLAN repair clears state["error"] after
    # dropping it, leaving an empty-but-legitimate plan for synthesize to explain honestly.
    assert body["error"] is None, f"{case.query!r} did not complete cleanly: {body['error']}"
    assert body["final_answer"], f"empty final_answer for {case.query!r}"
    assert isinstance(body["latency"], dict) and body["latency"]
    # every tool that actually ran recorded a real, non-negative latency (never fabricated)
    for tool_name in body["tool_results"]:
        assert body["latency"].get(f"tool:{tool_name}", 0) >= 0
