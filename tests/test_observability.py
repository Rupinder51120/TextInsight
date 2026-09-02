"""Observability — structured logging + /metrics (item 3 of the 2026-09-02 scope revision). Structlog's
actual JSON output isn't asserted here (that's exercised manually against a running server — see
docs/TECH_STACK.md); this covers the parts with real logic: MetricsRegistry's aggregation, and that
GET /metrics reflects real request traffic end-to-end through the FastAPI app.
"""

from fastapi.testclient import TestClient

from backend.main import app
from observability.metrics import MetricsRegistry


class TestMetricsRegistry:
    def test_snapshot_on_no_traffic_is_all_zero(self):
        registry = MetricsRegistry()

        snapshot = registry.snapshot()

        assert snapshot == {"request_count": 0, "error_count": 0, "average_latency_ms": 0.0, "error_rate": 0.0}

    def test_snapshot_reflects_recorded_requests(self):
        registry = MetricsRegistry()

        registry.record_request(duration_ms=100.0, is_error=False)
        registry.record_request(duration_ms=200.0, is_error=True)

        snapshot = registry.snapshot()

        assert snapshot["request_count"] == 2
        assert snapshot["error_count"] == 1
        assert snapshot["average_latency_ms"] == 150.0
        assert snapshot["error_rate"] == 0.5


class TestMetricsEndpoint:
    def test_metrics_endpoint_reflects_real_request_traffic(self):
        client = TestClient(app)

        before = client.get("/metrics").json()
        client.get("/session/does-not-exist/history")  # a cheap, deterministic 404
        after = client.get("/metrics").json()

        # >= 2, not exactly +2: the /metrics GET calls themselves are also counted.
        assert after["request_count"] >= before["request_count"] + 2
        assert after["error_count"] >= before["error_count"] + 1
