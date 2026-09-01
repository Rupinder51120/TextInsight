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

OutputT = TypeVar("OutputT", bound=BaseModel)


def timed_tool(fn: Callable[..., OutputT]) -> Callable[..., OutputT]:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> OutputT:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        result.latency_ms = (time.perf_counter() - start) * 1000
        return result

    return wrapper
