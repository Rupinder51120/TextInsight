"""@timed_tool — the tool-side counterpart to agent/timing.py's @timed.

Tool functions take/return Pydantic models, not AgentState (they must be independently callable/testable
outside the agent graph per docs/ARCHITECTURE.md §2), so timing is recorded on the Output model's
`latency_ms` field instead of directly into AgentState.latency. Day 3's `execute_tool` node reads this value
back out and folds it into AgentState.latency when a tool runs inside a real turn — see
docs/LATENCY_AND_PERFORMANCE.md §3.
"""

import functools
import time
from typing import Callable, TypeVar

from pydantic import BaseModel

from observability.logging import get_logger

OutputT = TypeVar("OutputT", bound=BaseModel)

_logger = get_logger("tool")


def timed_tool(fn: Callable[..., OutputT]) -> Callable[..., OutputT]:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> OutputT:
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            _logger.error(
                "tool_execution", tool=fn.__name__, duration_ms=round(duration_ms, 2), success=False, error=str(exc)
            )
            raise
        duration_ms = (time.perf_counter() - start) * 1000
        result.latency_ms = duration_ms
        _logger.info("tool_execution", tool=fn.__name__, duration_ms=round(duration_ms, 2), success=True)
        return result

    return wrapper
