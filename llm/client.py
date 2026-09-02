"""LLMClient — the only interface any agent node/tool uses to talk to an LLM.

Per docs/API_AND_SERVICES.md §5 and CLAUDE.md §4 ("No provider SDK imported outside its adapter"):
`langchain_groq` is imported ONLY in this file. Every other module depends on `LLMClient`, never on
`ChatGroq` directly — swapping providers later means adding one new adapter class here, not touching the
agent graph.
"""

import time
from abc import ABC, abstractmethod
from typing import Any

from config import settings
from observability.logging import get_logger

_logger = get_logger("llm")


class LLMError(RuntimeError):
    """Raised after retries are exhausted or on a non-retryable provider failure."""


class LLMClient(ABC):
    """Thin provider-agnostic interface. See docs/ARCHITECTURE.md §2."""

    @abstractmethod
    def complete(self, messages: list[dict[str, str]], context: str = "unspecified") -> str:
        """Send a list of {role, content} messages, return the assistant's text response.

        `context` is a caller-supplied label (e.g. the calling graph node or tool name) used only for
        structured logging (item 3 of the 2026-09-02 scope revision) — it has no effect on the request
        sent to the provider.
        """

    @abstractmethod
    def bind_tools(self, tools: list[Any]) -> "LLMClient":
        """Return a variant of this client with tool-calling schemas bound (used from Day 3 onward)."""


class GroqLLMClient(LLMClient):
    """Groq adapter, via langchain_groq.ChatGroq.

    Retry policy per docs/SECURITY_AND_RELIABILITY.md §6-7: bounded retries with exponential backoff on
    retryable errors only (timeouts, 429, 5xx) — never retried on 4xx client errors like a bad API key.
    """

    _MAX_RETRIES = 3
    _BACKOFF_BASE_SECONDS = 1.0

    def __init__(self, model: str | None = None, bound_tools: list[Any] | None = None):
        from langchain_groq import ChatGroq  # local import: keeps this the sole import site

        if not settings.groq_api_key:
            raise LLMError("GROQ_API_KEY is not set — see .env.example.")

        self._chat = ChatGroq(
            model=model or settings.groq_model,
            api_key=settings.groq_api_key,
            timeout=settings.llm_timeout_seconds,
            # The underlying groq SDK defaults to its OWN max_retries=2, independent of and invisible to
            # complete()'s retry loop below — and on a 429 it sleeps for Groq's exact stated Retry-After
            # value (up to 60s) per attempt, not our exponential backoff. Left at its default, a single
            # complete() call could silently block for minutes (3 outer attempts x up to 2 inner SDK
            # retries x up to 60s), which directly undermines the bounded-retry, fail-fast-to-degraded-mode
            # policy this class exists to implement (docs/SECURITY_AND_RELIABILITY.md §6-7). Disabling it
            # here makes complete()'s own loop the single, predictable source of retry behavior.
            max_retries=0,
        )
        if bound_tools:
            self._chat = self._chat.bind_tools(bound_tools)

    def complete(self, messages: list[dict[str, str]], context: str = "unspecified") -> str:
        start = time.perf_counter()
        last_exc: Exception | None = None
        for attempt in range(self._MAX_RETRIES):
            try:
                response = self._chat.invoke(messages)
                duration_ms = (time.perf_counter() - start) * 1000
                _logger.info(
                    "llm_call",
                    node=context,
                    model=settings.groq_model,
                    duration_ms=round(duration_ms, 2),
                    success=True,
                    attempt=attempt + 1,
                )
                return response.content
            except Exception as exc:  # noqa: BLE001 — provider errors are inspected below, not swallowed
                last_exc = exc
                if not self._is_retryable(exc) or attempt == self._MAX_RETRIES - 1:
                    duration_ms = (time.perf_counter() - start) * 1000
                    _logger.error(
                        "llm_call",
                        node=context,
                        model=settings.groq_model,
                        duration_ms=round(duration_ms, 2),
                        success=False,
                        error=str(exc),
                        attempt=attempt + 1,
                    )
                    raise LLMError(f"Groq call failed: {exc}") from exc
                time.sleep(self._BACKOFF_BASE_SECONDS * (2**attempt))
        raise LLMError(f"Groq call failed after {self._MAX_RETRIES} attempts: {last_exc}")

    def bind_tools(self, tools: list[Any]) -> "GroqLLMClient":
        return GroqLLMClient(
            model=self._chat.model_name if hasattr(self._chat, "model_name") else None, bound_tools=tools
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            return status_code == 429 or status_code >= 500
        # Timeouts and connection errors surface as plain TimeoutError/OSError from the HTTP layer.
        return isinstance(exc, (TimeoutError, ConnectionError))
