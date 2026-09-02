"""In-process request metrics for GET /metrics (item 3 of the 2026-09-02 scope revision).

Deliberately not Prometheus/a metrics backend — CLAUDE.md §3.5's "no external service beyond what's asked"
boundary still applies; a simple in-process counter behind a JSON endpoint is the documented scope here.

Same encapsulated-registry pattern as `models/registry.py`'s pipeline cache: a thread-safe, process-wide
counter, not per-session/per-request state, so this is not the module-level-mutable-`df` anti-pattern
CLAUDE.md §4 warns against — nothing here is request-shaped data, only aggregate counts.
"""

import threading


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count = 0
        self._error_count = 0
        self._total_latency_ms = 0.0

    def record_request(self, duration_ms: float, is_error: bool) -> None:
        with self._lock:
            self._request_count += 1
            self._total_latency_ms += duration_ms
            if is_error:
                self._error_count += 1

    def snapshot(self) -> dict:
        with self._lock:
            count = self._request_count
            total_latency_ms = self._total_latency_ms
            error_count = self._error_count

        average_latency_ms = (total_latency_ms / count) if count else 0.0
        error_rate = (error_count / count) if count else 0.0
        return {
            "request_count": count,
            "error_count": error_count,
            "average_latency_ms": round(average_latency_ms, 2),
            "error_rate": round(error_rate, 4),
        }


metrics_registry = MetricsRegistry()
