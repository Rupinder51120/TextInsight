"""Structured logging setup — structlog, JSON output (item 3 of the 2026-09-02 scope revision, see
CLAUDE.md §3.5 / docs/TECH_STACK.md). Replaces ad-hoc print/logging with one configured, structured logger
used everywhere: `tools/timing.py` (every tool execution), `llm/client.py` (every LLM call), and
`backend/main.py` (every request) all call `get_logger` from here rather than configuring their own.

Configured once, at import time, the same eager-init pattern `config.py` already uses for `settings` — safe
to import this module from multiple places since `structlog.configure` just (re)sets the global processor
chain.
"""

import logging
import sys

import structlog


def configure_logging(level: int = logging.INFO) -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


configure_logging()
