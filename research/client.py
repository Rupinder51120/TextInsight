"""ResearchClient — the only interface any tool uses for web research.

Per docs/API_AND_SERVICES.md §3 and §5: implemented today by a Tavily adapter; a future adapter
(Serper/Bing/etc.) implements the same search() interface. No tool imports `tavily` directly — matches the
same provider-abstraction philosophy as llm/client.py's LLMClient/GroqLLMClient.
"""

from abc import ABC, abstractmethod

from pydantic import BaseModel

from config import settings


class Evidence(BaseModel):
    title: str
    url: str
    snippet: str


class ResearchError(RuntimeError):
    """Raised on a genuine search-API failure. Callers (tools/research_models.py) are expected to catch
    this and degrade to found: false — docs/API_AND_SERVICES.md §3: "On failure/quota exhaustion,
    research_models returns found: false ... never silently omitted"."""


class ResearchClient(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        """Returns up to max_results attributed evidence items for query."""


class TavilyResearchClient(ResearchClient):
    def __init__(self):
        if not settings.tavily_api_key:
            raise ResearchError("TAVILY_API_KEY is not set — see .env.example.")

        from tavily import TavilyClient  # local import: keeps this the sole import site

        self._client = TavilyClient(api_key=settings.tavily_api_key)

    def search(self, query: str, max_results: int = 5) -> list[Evidence]:
        try:
            response = self._client.search(query=query, max_results=max_results)
        except Exception as exc:  # noqa: BLE001 — any Tavily/network failure degrades the same way
            raise ResearchError(f"Tavily search failed: {exc}") from exc

        return [
            Evidence(
                title=r.get("title", "") or "(untitled)",
                url=r.get("url", ""),
                snippet=(r.get("content", "") or "")[:500],
            )
            for r in response.get("results", [])
        ]
