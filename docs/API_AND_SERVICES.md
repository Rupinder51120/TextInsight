# API_AND_SERVICES.md — TextInsight

## 1. Groq API

- **Used for**: `understand_intent`, `plan_steps`, `synthesize`/`explain_results`, and the reasoning portion
  of `model_recommendation`. Never used for the deterministic NLP outputs (sentiment/NER/classification/
  summarization/embeddings), which run locally via Hugging Face.
- **Access pattern**: via `langchain_groq.ChatGroq`, wrapped behind the project's own `LLMClient` interface
  so no other module imports `langchain_groq` directly.
- **Model choice**: a fast instruction-following/tool-calling-capable model available on Groq at build time
  (e.g., a Llama 3.x instant/versatile variant); exact model id is a config value (`GROQ_MODEL`), not
  hardcoded, so it can be changed without code edits as Groq's catalog changes.
- **Env vars**: `GROQ_API_KEY` (required), `GROQ_MODEL` (optional, has a default), `LLM_TIMEOUT_SECONDS`
  (optional, has a default).
- **Rate limits / fallback**: Groq's free/dev tier has request-per-minute and token-per-minute limits (exact
  numbers to be confirmed against Groq's current published limits at build time — not invented here). The
  `LLMClient` wraps calls with: (a) a timeout, (b) bounded retry with exponential backoff on 429/5xx, (c) a
  final graceful failure path — `synthesize` falls back to a templated summary, `model_recommendation` falls
  back to rule-based-only output, `understand_intent` failure surfaces a clear "please rephrase / try again"
  error rather than a crash.
- **Secrets**: `GROQ_API_KEY` read from environment / `.env` (via `python-dotenv` or Pydantic `Settings`),
  never logged, never returned in any API response, never committed to source control (`.env` in
  `.gitignore`, `.env.example` provided with placeholder values).

## 2. Hugging Face Usage

- **Used for**: local model weights for sentiment, zero-shot classification, NER, summarization
  (`transformers`), and embeddings (`sentence-transformers`).
- **Access pattern**: models pulled from the Hugging Face Hub on first run and cached locally
  (`HF_HOME`/default cache dir); no Hugging Face Inference API calls in the default configuration — all
  inference is local, which is a deliberate latency/cost/reliability choice.
- **Env vars**: `HF_HOME` (optional, override cache location), `HF_HUB_OFFLINE` (optional, for fully offline
  demo runs once models are cached).
- **Secrets**: none required for public model downloads; an optional `HF_TOKEN` may be added later only if a
  gated model is ever introduced (not planned for the default model list in `TOOLS_AND_MODELS.md`).
- **Rate limits / fallback**: Hub download rate limits are a one-time, first-run concern; if a model fails to
  download (offline environment, network block), the affected tool reports a clear "model unavailable"
  error rather than silently degrading to a different model.

## 3. Research / Search API (Tavily default)

- **Used for**: `research_models` tool only.
- **Access pattern**: single search endpoint call per research request, constrained to a small number of
  results (e.g., top 5), each returned with title/url/snippet.
- **Env vars**: `TAVILY_API_KEY` (or `SEARCH_API_KEY` if a different provider is substituted — the
  `ResearchClient` interface is provider-agnostic, matching the same abstraction philosophy as the LLM
  client).
- **Rate limits / fallback**: free-tier query limits apply (exact current limits to be confirmed against the
  provider's published docs at build time). On failure/quota exhaustion, `research_models` returns
  `found: false` and the agent proceeds with a system-judgment-only recommendation, explicitly labeled as
  such in the response — never silently omitted.
- **Secrets**: same handling as `GROQ_API_KEY` — env-only, never logged/returned/committed.
- **Reliability note**: search results are treated as untrusted external content — snippets are quoted
  narrowly and attributed, never treated as instructions to the agent (no prompt-injection risk accepted
  from search results; see `SECURITY_AND_RELIABILITY.md`).

## 4. LangSmith (optional)

- **Used for**: tracing LangGraph execution during development/demo.
- **Access pattern**: enabled purely via environment variables recognized by LangChain/LangGraph
  (`LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`); no code branches on its presence.
- **Env vars**: `LANGCHAIN_TRACING_V2` (`true`/`false`, default `false`), `LANGCHAIN_API_KEY` (optional),
  `LANGCHAIN_PROJECT` (optional).
- **Rate limits / fallback**: irrelevant to core functionality — if unset or the API is unreachable, tracing
  simply doesn't occur; the agent must not depend on LangSmith being up.
- **Secrets**: same handling as other API keys.

## 5. Provider Abstraction Summary

Two abstraction seams exist, both deliberately thin:

- `LLMClient` — implemented today by a Groq adapter; future adapters (OpenAI/Anthropic/local) implement the
  same `complete()`/tool-calling interface.
- `ResearchClient` — implemented today by a Tavily adapter; future adapters (Serper/Bing/etc.) implement the
  same `search(query) -> list[Evidence]` interface.

No node in the LangGraph agent, and no tool, imports a provider SDK directly — everything goes through these
two interfaces. This is what makes "Groq can be replaced later without rewriting the application" true rather
than aspirational.

## 6. Environment Variables (consolidated)

```
GROQ_API_KEY=            # required
GROQ_MODEL=               # optional, default set in config
LLM_TIMEOUT_SECONDS=      # optional, default set in config
TAVILY_API_KEY=           # optional; research degrades gracefully if unset
HF_HOME=                  # optional
HF_HUB_OFFLINE=           # optional
LANGCHAIN_TRACING_V2=     # optional, default false
LANGCHAIN_API_KEY=        # optional
LANGCHAIN_PROJECT=        # optional
MAX_UPLOAD_MB=            # optional, default set in config
```

An `.env.example` file with these keys (placeholder values, no real secrets) should be part of the delivered
repo.
