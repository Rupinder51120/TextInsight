"""Centralized app configuration via Pydantic Settings.

Every env var listed in docs/API_AND_SERVICES.md §6 has a field here. No module outside this
file reads os.environ directly for these values.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Groq / LLM (docs/API_AND_SERVICES.md §1)
    # Default verified live against Groq's /models endpoint on 2026-09-01 — the docs' example
    # ("a Llama 3.x instant/versatile variant") is no longer in Groq's catalog as of this date.
    # See docs/API_AND_SERVICES.md §1: "exact model id is a config value, not hardcoded" — override via
    # GROQ_MODEL if Groq's catalog changes again.
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-20b"
    llm_timeout_seconds: float = 30.0

    # Research (docs/API_AND_SERVICES.md §3) — optional, research degrades gracefully if unset
    tavily_api_key: str = ""

    # Hugging Face (docs/API_AND_SERVICES.md §2)
    hf_home: str = ""
    hf_hub_offline: bool = False

    # LangSmith tracing (docs/API_AND_SERVICES.md §4) — optional, must never be a hard dependency
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = ""

    # Uploads (docs/SECURITY_AND_RELIABILITY.md §1, §5)
    max_upload_mb: float = 20.0
    max_rows: int = 5000
    max_pdf_pages: int = 200

    # Agent (docs/ARCHITECTURE.md §3.3)
    max_tool_iterations: int = 6


settings = Settings()
