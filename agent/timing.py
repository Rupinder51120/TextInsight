"""@timed — wraps a graph node (or, from Day 2, a tool function) to record its duration.

Per docs/LATENCY_AND_PERFORMANCE.md §3: every tool and every LangGraph node writes its duration into
AgentState.latency, keyed by name. Never fabricated (CLAUDE.md §5) — this is the only place a duration
value is produced, always from an actual wall-clock measurement around the wrapped call.
"""

import functools
import time
from typing import Callable

from agent.state import AgentState


def timed(name: str) -> Callable[[Callable[[AgentState], AgentState]], Callable[[AgentState], AgentState]]:
    def decorator(fn: Callable[[AgentState], AgentState]) -> Callable[[AgentState], AgentState]:
        @functools.wraps(fn)
        def wrapper(state: AgentState) -> AgentState:
            start = time.perf_counter()
            result = fn(state)
            duration_ms = (time.perf_counter() - start) * 1000
            result["latency"] = {**result.get("latency", {}), name: duration_ms}
            return result

        return wrapper

    return decorator
