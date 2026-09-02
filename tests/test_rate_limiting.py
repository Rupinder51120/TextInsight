"""Rate limiting on /query (item 4 of the 2026-09-02 scope revision) — protects this API from being
hammered by one client, separate from llm/client.py's Groq-provider-side retry/backoff handling.

slowapi's limiter is a shared, process-wide, in-memory bucket keyed by client IP (backend/main.py's
`limiter`) — not reset between tests, so this asserts "eventually 429s", not an exact request count, and
runs against a fixture-uploaded session that never reaches the agent (a nonexistent session_id 404s before
the limiter's own state matters, keeping this fast and independent of Groq).
"""

from fastapi.testclient import TestClient

from backend.main import app


def test_query_endpoint_eventually_rate_limits_a_single_client():
    client = TestClient(app)

    statuses = [
        client.post("/query", json={"session_id": "does-not-exist", "query": "hi"}).status_code for _ in range(15)
    ]

    assert 429 in statuses, f"expected a 429 among 15 rapid requests, got statuses: {statuses}"
    # Every response before the first 429 should be the endpoint's normal validation-error path (404 for
    # an unknown session), not silently swallowed by the limiter.
    first_429 = statuses.index(429)
    assert all(status == 404 for status in statuses[:first_429])
