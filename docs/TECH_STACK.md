# TECH_STACK.md — TextInsight

For each technology: role, why chosen, alternatives considered, execution mode, GPU/resource notes.

## Language: Python 3.11+
- **Role**: sole implementation language, backend/agent/tools/frontend.
- **Why**: native ecosystem for LangGraph/LangChain, Hugging Face, PyTorch, FastAPI, Streamlit; matches
  Claude Code's strongest tooling; single-language project is easier to build and explain in 5 days.
- **Alternatives**: TypeScript (rejected — fragments the NLP/ML ecosystem, no benefit here).
- **Execution**: local process (dev) or single container (deploy). No GPU required for the language itself.

## Agent: LangGraph + LangChain
- **Role**: LangGraph provides the stateful graph (nodes/edges/conditional routing) that is the literal
  agentic core; LangChain provides tool-binding utilities, prompt templates, and the LLM abstraction layer
  used underneath LangGraph.
- **Why**: LangGraph is explicitly required and is the right tool for *stateful, multi-step, conditionally
  routed* execution — a plain function-calling loop could approximate it but wouldn't demonstrate explicit
  state/edges/conditional routing as clearly for a portfolio review.
- **Alternatives**: hand-rolled agent loop (rejected — reinvents state machine handling; less legible to
  reviewers who know LangGraph); CrewAI/AutoGen (rejected — oriented around multi-agent chat, unnecessary
  complexity for a single-agent tool-router, explicitly excluded by scope).
- **Execution**: in-process Python library, no external service.

## LLM: Groq (via provider-abstraction interface)
- **Role**: powers `understand_intent`, `plan_steps` (or combined), `synthesize`, and `model_recommendation`'s
  reasoning step. Never used for the deterministic NLP outputs themselves.
- **Why**: only available API key today; Groq's hosted inference is fast (a real advantage for a
  latency-focused project), and it exposes OpenAI-compatible-style chat/tool-calling that fits LangChain's
  `ChatGroq` integration.
- **Alternatives**: OpenAI (not available — no key), local LLM via Ollama (rejected for MVP — adds setup
  complexity/latency variance; can be a future provider adapter), Anthropic API (viable future adapter,
  same reason as OpenAI — abstraction is built so this is a follow-up, not a rewrite).
- **Provider abstraction**: a single `LLMClient` interface (thin wrapper around LangChain's chat-model
  interface) is the only place that imports `langchain_groq`; every other module depends on the interface,
  not the provider. Swapping providers means adding one adapter + one config value.
- **Execution**: hosted API, no local compute.

## NLP: Hugging Face Transformers + PyTorch + Sentence-Transformers
- **Role**: the deterministic inference layer — sentiment, zero-shot classification, NER, summarization
  (Transformers `pipeline`s) and embeddings (Sentence-Transformers) for semantic search.
- **Why**: mature, well-documented pretrained models with predictable latency/quality tradeoffs; keeps NLP
  results reproducible and separable from LLM reasoning, which matters for the project's "don't fabricate,
  don't blur measured-vs-claimed" principle.
- **Alternatives**: spaCy for NER (viable, smaller footprint — noted as an alternative in
  `TOOLS_AND_MODELS.md`, not chosen as default to keep one dependency family); doing everything through the
  LLM (rejected — non-deterministic, harder to test, conflates "understanding" with "inference," against
  the project's explicit principle to prefer deterministic code for deterministic tasks).
- **Execution**: local, in-process. **GPU**: optional accelerator only; every default model choice in
  `TOOLS_AND_MODELS.md` is chosen to be CPU-tractable for the input sizes in scope (short-to-medium text,
  moderate batch sizes). CUDA is auto-used if available via PyTorch, not required.

## Data: Pandas + NumPy
- **Role**: CSV parsing, dataframe-based profiling and filtering, numeric handling for stats/scores.
- **Why**: standard, no reason to deviate; matches "reuse robust libraries, avoid unnecessary complexity."
- **Alternatives**: Polars (faster, but adds a second dataframe API to reason about for no scope benefit at
  the data sizes in this project — rejected).
- **Execution**: local, CPU, in-process.

## Semantic Search: FAISS
- **Role**: vector index for `generate_embeddings` / `semantic_search`.
- **Why**: local, fast, no external service dependency, well-suited to per-session, moderate-size (thousands
  of vectors) indices; keeps semantic search latency low and fully under our control/measurable.
- **Alternatives**: a managed vector DB (Pinecone/Weaviate/Chroma-as-service) — rejected as unnecessary
  infrastructure for session-scoped, moderate-size corpora (explicitly against "no unnecessary
  microservices"); Chroma (viable lightweight alternative, similar footprint to FAISS — FAISS chosen because
  it was the stated preference and is simpler to reason about for exact/approximate KNN at this scale).
- **Execution**: local, in-process, CPU. Index persisted to disk per session.

## PDF: PyMuPDF (fitz)
- **Role**: text extraction from uploaded PDFs during ingestion.
- **Why**: fast, dependency-light, handles the common case (text-based PDFs) well without extra services.
- **Alternatives**: `pdfplumber` (slower, more table-aware — not needed since target is prose documents, not
  tables); OCR pipelines (out of scope — scanned/image PDFs are explicitly not a target input).
- **Execution**: local, CPU, in-process.

## Backend: FastAPI + Uvicorn
- **Role**: HTTP API surface between Streamlit and the LangGraph agent; owns request validation and session
  routing.
- **Why**: async-friendly, native Pydantic integration (matches the validation requirement), minimal
  boilerplate, standard choice for a Python service in a 5-day build.
- **Alternatives**: Flask (less native async/typing support), Django (far more than needed — explicitly
  against over-engineering).
- **Execution**: local process (`uvicorn app:app`), single instance for the project's scope.

## Frontend: Streamlit
- **Role**: file upload, chat-style query box, workflow/status display, result rendering (tables, charts via
  `st.plotly_chart`/`st.bar_chart`), latency panel.
- **Why**: explicit preference; fastest way to build a genuinely usable, good-looking demo UI in Python
  within 5 days, with first-class file-upload and dataframe/chart widgets.
- **Alternatives**: React frontend — rejected per explicit instruction unless it provides a *meaningful*
  benefit; it does not here (no need for complex client-side state, custom interactivity, or multi-page
  routing beyond what Streamlit provides), and it would consume build-time better spent on the agent.
- **Execution**: local process, talks to FastAPI over HTTP (kept as a separate process/service, not
  merged into the backend, so the API remains independently usable/testable — e.g., via curl or automated
  tests — matching the "backend + frontend" requirement literally).

## Validation: Pydantic
- **Role**: request/response schemas for FastAPI, and input/output schemas for every tool (used both for
  validation and as the LLM tool-calling schema source via LangChain).
- **Why**: single validation library reused across the whole stack; matches "keep components modular" and
  gives the agent's tool contracts real type safety.
- **Alternatives**: manual dict validation (rejected — error-prone, no LLM tool-schema generation benefit).

## Observability: LangSmith (optional)
- **Role**: trace LangGraph runs (node sequence, tool calls, latencies, LLM prompts/responses) during
  development and demoing.
- **Why**: pairs natively with LangGraph, gives visual proof of the agent's actual multi-step behavior —
  valuable both for debugging and for showing interviewers a real trace, not just a claim.
- **Constraint**: must be **optional** — the system must run correctly with `LANGCHAIN_TRACING_V2=false`/no
  API key set (see `API_AND_SERVICES.md`), so a reviewer without a LangSmith account can still run the
  project.
- **Alternatives**: no tracing at all (would weaken the "prove it's actually agentic" story); a custom
  logging solution (more work for less insight than LangSmith gives for free).

## Research: Web Search API
- **Role**: backs the `research_models` tool.
- **Recommended default**: **Tavily Search API** — purpose-built for LLM/agent use (returns clean,
  summarized, source-attributed results rather than raw HTML), has a free tier suitable for a portfolio
  project, and has first-class LangChain integration.
- **Alternatives**: Serper.dev / SerpAPI (Google-results wrappers, more raw HTML/snippet handling needed);
  Bing Search API (viable, heavier setup). Any of these fit behind the same thin `ResearchClient` interface
  used by `research_models`, so the choice is not architecturally load-bearing — Tavily is the default
  because it minimizes extraction work within the 5-day budget.
- **Execution**: hosted API, no local compute.

## Deployment: Docker + Docker Compose
- **Role**: containerizes the FastAPI backend and Streamlit frontend as two services, orchestrated via a
  single `docker-compose.yml`, so the whole app runs with one command on any machine without a manual
  Python environment setup.
- **Why**: makes the project trivially runnable by anyone reviewing it (`docker compose up`), removes
  "works on my machine" risk from Hugging Face/PyTorch version mismatches, and is a standard, expected
  piece of deployment maturity on a portfolio project.
- **Structure**:
  - `backend/Dockerfile` — Python base image, installs `requirements.txt`, runs `uvicorn`.
  - `frontend/Dockerfile` — Python base image, installs `requirements.txt`, runs `streamlit run`.
  - `docker-compose.yml` — defines both services, a shared network so Streamlit can reach FastAPI by
    service name, environment variables passed through from a root `.env` (never baked into the image),
    and a named volume for the Hugging Face model cache so models aren't re-downloaded on every container
    rebuild.
- **What Docker is NOT used for here**: no database container, no cache/Redis container, no message queue —
  per the system-design decisions in `PROJECT_SPEC.md`, none of those exist in this project, so compose
  stays to exactly two services.
- **Alternatives**: running both processes directly with `uvicorn`/`streamlit run` (still fully supported
  and remains the fastest loop for active development — Docker is for demoing/sharing/deploying, not for
  the primary dev inner loop) — see `README_PLAN.md` for both paths documented side by side.
- **Local vs cloud**: Docker images are built and run locally for this project's scope; no cloud
  deployment (ECS/Cloud Run/etc.) is in scope, though the same images would be the starting point for one
  later.
- **GPU note**: no GPU passthrough configured — matches the project's CPU-only deployability requirement.

## Cross-Cutting Notes

- **Local vs API/cloud summary**: LLM (Groq) and web research (Tavily) are the only external network
  dependencies at inference time; all NLP inference, embeddings, semantic search, ingestion, and validation
  are local/in-process, keeping the core latency story mostly under our control.
- **GPU requirement**: none. The project must be demoable on a CPU-only laptop; GPU, if present, is used
  automatically by PyTorch but is never assumed.
- **Resource footprint**: default models are chosen (see `TOOLS_AND_MODELS.md`) to keep combined memory
  footprint reasonable on a typical developer laptop (roughly low single-digit GB across loaded pipelines).
