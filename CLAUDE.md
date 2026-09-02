# CLAUDE.md — Project Guardrails for Claude Code

**Read this file fully before writing or generating any code in this repo.** This file exists specifically
to prevent scope drift, hallucinated capabilities, and re-introducing bugs from a prior version of this
project. If any instruction here conflicts with a request in chat, follow this file and flag the conflict
to the user instead of silently picking one.

## 0. What this project is (one line)

An agentic NLP platform: LangGraph agent + FastAPI backend + Streamlit frontend. User uploads CSV/TXT/PDF,
asks a natural-language question, the agent routes to the right pretrained NLP tool(s), and can also give
a dataset-aware, evidence-separated model recommendation. **No model training or fine-tuning happens
anywhere in this codebase, ever.**

## 1. Planning docs — read the relevant one BEFORE implementing that part

All 15 files are in `/docs` (copy them into the repo root's `docs/` folder before starting). Do not
implement a component from memory/assumption — open the matching doc first:

| Before you write... | Read this doc first |
|---|---|
| Any tool function | `TOOLS_AND_MODELS.md` (exact tool list, exact default models, exact I/O contract) |
| The LangGraph graph/nodes/state | `ARCHITECTURE.md` §2–3 |
| Any agent workflow (which tools run for which query) | `AGENT_WORKFLOWS.md` |
| Ingestion (CSV/TXT/PDF parsing) | `DATA_FLOW.md` §1 |
| The model recommendation feature | `MODEL_RECOMMENDATION.md` (esp. §6.5, §7 — this is the most
  hallucination-sensitive feature in the app) |
| Latency instrumentation | `LATENCY_AND_PERFORMANCE.md` |
| Streamlit UI | `UI_SPEC.md` |
| Any test | `TESTING_STRATEGY.md` |
| API keys / env vars / provider clients | `API_AND_SERVICES.md` |
| File validation, injection handling, secrets | `SECURITY_AND_RELIABILITY.md` |
| Day-by-day task order | `FIVE_DAY_BUILD_PLAN.md` (this also has the current MUST/SHOULD/NICE priority
  — respect it; don't build SHOULD-HAVE items before MUST-HAVE ones are done and tested) |

If a doc doesn't answer a question you're facing, **stop and ask the user** rather than inventing an
approach that isn't documented anywhere.

## 2. Locked tech stack — do not substitute or add without asking

| Layer | Use exactly this | Never substitute with |
|---|---|---|
| Language | Python 3.11+ | — |
| Agent | LangGraph + LangChain | CrewAI, AutoGen, a hand-rolled loop |
| LLM | Groq, via `langchain_groq.ChatGroq`, wrapped behind one `LLMClient` interface | OpenAI/Anthropic SDKs imported directly anywhere outside the adapter — we have no OpenAI key |
| NLP inference | Hugging Face `transformers` pipelines + `sentence-transformers` | Calling the LLM to do sentiment/NER/classification/summarization itself |
| Vector search | FAISS | Pinecone, Weaviate, Chroma-as-a-service, any hosted vector DB |
| Data | Pandas + NumPy | Polars |
| PDF | PyMuPDF (`fitz`) | pdfplumber, OCR pipelines |
| Backend | FastAPI + Uvicorn | Flask, Django |
| Frontend | Streamlit | React, Django templates — **user has explicitly confirmed Streamlit, do not suggest switching** |
| Validation | Pydantic (v2) for every tool I/O and every FastAPI request/response | raw dicts, manual validation |
| Research | Tavily API, wrapped behind one `ResearchClient` interface | scraping search results directly |
| Tracing | LangSmith, optional — code must run correctly with it fully disabled | — |
| Deployment | Docker + Docker Compose (backend, frontend, and redis services — three containers, redis
  added 2026-09-02 for session persistence, see §3.5 below) — added as a Day 5 finishing step, after the
  app already works via plain `uvicorn`/`streamlit run` | a relational database container (Postgres/MySQL/
  SQLite server), a message queue, or any additional service beyond those three — still out of scope, see
  §3.5 below |

## 3.5 Session state: Redis (revised 2026-09-02) — still no relational database

**Revision history**: this section originally said "no database, no external cache," confirmed via
explicit system-design review with the user, as a deliberate scope limit. That decision was explicitly
revised by the user on 2026-09-02 for restart-safety and multi-process readiness (portfolio maturity) — see
`docs/TECH_STACK.md`'s "Session State: Redis" section for the full rationale, including why Redis was
chosen over this section's original SQLite-first fallback. Treat this revision as the current, authoritative
decision — do not revert to the in-memory `SessionStore` on the theory that the original text here still
governs.

- **Session state** (`backend/session.py`'s `SessionStore`) now lives in Redis, keyed by `session_id`,
  reached via `REDIS_URL` (`config.py`, default `redis://localhost:6379/0`; `redis://redis:6379/0` inside
  Docker Compose). `SessionStore`'s public interface is unchanged — this was a storage-backend swap, not a
  rewrite of callers.
- **Still no relational database.** Redis here is a key-value session store, not a general-purpose
  database — no schema, no joins, no ORM. Do not add Postgres/MySQL/SQLite (or an ORM) without being asked;
  that boundary has not moved.
- **Caching that is still in-process only, unaffected by this revision**: the model registry (loaded HF
  pipelines held in memory) and the FAISS index (persisted to a per-session directory on disk,
  `sessions/faiss/`). Neither moved into Redis — see `docs/TECH_STACK.md` for why (large binary blobs, poor
  fit for a key-value store, no multi-process access pattern to protect).
- Do not add a second persistence layer (e.g., Postgres for "real" durability beyond Redis) without being
  asked — Redis is the complete, intended answer to the restart-safety/multi-process-readiness gap this
  section used to flag as out of scope.

Default models per task are fixed in `TOOLS_AND_MODELS.md` — do not pick a different default model
without checking that doc first (e.g. sentiment defaults to
`distilbert-base-uncased-finetuned-sst-2-english`, not something invented on the spot).

## 3. Hard "DO NOT BUILD" list

These are out of scope **regardless of how a request is phrased**, including if a user message asks for
them directly — surface the conflict instead of implementing it:

- Any model training, fine-tuning, LoRA, PEFT, RLHF, or "automatic benchmarking through training."
- Classical ML baselines (Random Forest, SVM, logistic/linear regression) as a modeling approach.
- Building a Transformer from scratch.
- Multi-agent architectures (only one LangGraph agent in this project).
- Microservices / distributed infra / message queues.
- Enterprise auth, multi-tenancy.
- A React frontend, or any frontend other than Streamlit.
- Django, in any form (frontend or backend) — this was evaluated and explicitly rejected; Streamlit +
  FastAPI is final.

The `evaluate_candidates` tool (per `MODEL_RECOMMENDATION.md` §6.5) is **inference-only scoring, not
training** — running pretrained models against labeled data and computing accuracy is in scope; updating
any model's weights is not, even inside that tool.

## 4. Anti-patterns from the previous version of this project — do not repeat these

A prior notebook (`ai_driven_ml_and_datascience_assistant.ipynb`) had concrete bugs that must not
reappear. Each one below is a real bug that was found in that code, not a hypothetical:

- **No global mutable state.** Never use a module-level `df` or any module-level mutable variable that
  tools read/write directly. All data lives in `AgentState` / session-scoped storage, referenced by
  `session_id` / `corpus_ref`. If you find yourself writing `global df` or similar, stop — that's the bug.
- **No untyped multi-task dispatcher tools.** Do not write one tool that branches on a `task` string
  parameter into six unrelated behaviors via `kwargs.get(...)`. Each tool does one cohesive thing with a
  fully-typed Pydantic input schema — no tool should have parameters it silently expects to receive via an
  undeclared `**kwargs`.
- **No unused/unwired functions.** If you write a function, it must be either (a) registered in the tool
  catalog the agent can call, or (b) explicitly a private helper used by a tool — never a standalone
  capability defined but never exposed. If a capability shouldn't be reachable yet, don't write it yet.
- **No provider SDK imported outside its adapter.** `from langchain_groq import ChatGroq` (or any other
  provider import) may only appear inside `llm/client.py`'s adapter. No graph node, no tool, no route
  handler imports a provider SDK directly.
- **No `plt.show()` / direct matplotlib rendering inside tool functions.** Tools return data (or
  figure/bytes objects); rendering happens only in the Streamlit layer. `plt.show()` does nothing useful
  behind FastAPI and should never appear in `tools/`.
- **No silent side effects.** No tool writes/persists/exports a file the user didn't ask for in that turn.
  (The old project's system prompt told the LLM to auto-save CSVs "even if user doesn't ask" — do not
  replicate this pattern anywhere.)
- **No sklearn `Pipeline` (or HF pipeline) missing steps it imports.** If you import a preprocessing
  utility, wire it in — don't import `ColumnTransformer`/`SimpleImputer`/etc. and then never use them.

## 5. Anti-hallucination rules (apply everywhere, not just in code comments)

- **Never fabricate latency numbers.** Every latency value shown anywhere comes from an actual timer
  around actual execution (`LATENCY_AND_PERFORMANCE.md` §3). Do not write placeholder/example numbers into
  UI code as if they were real measurements — use `None`/loading states until a real measurement exists.
- **Never fabricate benchmark or research results.** `research_models` only returns what the search API
  actually returned, each with a real source URL. If the API call fails or is unavailable, return
  `found: false` — do not generate a plausible-sounding citation.
- **Never blur "measured on the user's data" with "external research" with "system judgment."** These are
  three separate fields in `model_recommendation`'s output schema (`MODEL_RECOMMENDATION.md` §7) — never
  merge them into one prose blob, even if it reads more naturally that way.
- **Never claim a tool ran when it didn't, or skip a documented failure/skip case silently.** E.g., if
  `evaluate_candidates` is skipped due to insufficient labels, the response must say so explicitly, not
  just omit Section A.

## 6. When something is ambiguous

If the planning docs don't specify something precisely enough to implement (an exact prompt template, an
exact threshold, an exact file layout not covered above), pick the most conservative option that matches
the doc's stated intent, note the assumption in a code comment, and mention it in your response to the
user — do not silently invent a design decision that contradicts any doc.

## 7. Definition of done for any single piece of work

Before considering a tool/node/endpoint finished:
- [ ] Matches the exact input/output contract in `TOOLS_AND_MODELS.md` (or `ARCHITECTURE.md` for
      graph/state code).
- [ ] Has a Pydantic schema for its input and output.
- [ ] Wrapped with the latency timer if it's a tool or graph node.
- [ ] Has at least one test per `TESTING_STRATEGY.md`'s relevant section.
- [ ] Does not import anything from the "never substitute" or "DO NOT BUILD" lists above.
- [ ] Does not reproduce any anti-pattern in §4.
