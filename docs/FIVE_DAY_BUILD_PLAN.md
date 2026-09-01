# FIVE_DAY_BUILD_PLAN.md — TextInsight

Assumes Claude Code as the primary implementation assistant, working from this planning package. Each day
lists files to create, functionality, dependencies, tests, and completion criteria.

## Day 1 — Foundation + Ingestion + LangGraph Skeleton

**Files**: `pyproject.toml`/`requirements.txt`, `config.py` (Pydantic `Settings`, env vars), `.env.example`,
`ingestion/` (`csv_loader.py`, `txt_loader.py`, `pdf_loader.py`, `corpus.py` — `Document`/`Corpus` models),
`agent/state.py` (`AgentState` TypedDict), `agent/graph.py` (skeleton graph with stub nodes — no tools yet,
just prove state flows through nodes/edges), `llm/client.py` (`LLMClient` interface + Groq adapter).

**Functionality**: parse CSV/TXT/PDF into a normalized `Corpus`; a minimal LangGraph graph that takes a query
and a corpus reference, calls the LLM once, and returns a response — proves the skeleton before any NLP
tools exist.

**Dependencies**: none beyond Day 1's own files.

**Tests**: ingestion unit tests (§ per format, including malformed input); `LLMClient` mocked test; a
smoke test that the skeleton graph runs end-to-end.

**Completion criteria**: `python -c "..."` or a small script can upload a sample CSV/TXT/PDF, get back a
normalized corpus, and get one LLM-generated response through the graph skeleton, with basic tests passing.

## Day 2 — NLP Tools

**Files**: `tools/profile_dataset.py`, `tools/sentiment_analysis.py`, `tools/text_classification.py`,
`tools/summarize_text.py`, `tools/schemas.py` (shared Pydantic input/output models), `models/registry.py`
(lazy-loaded, cached HF pipeline instances). `named_entity_recognition.py` is written only if time remains
at the end of the day (SHOULD-HAVE, not MUST-HAVE — see `PROJECT_SPEC.md` §11 for why it was deprioritized).

**Functionality**: the four MUST-HAVE deterministic single-tool capabilities working and independently
callable/testable, using the default models from `TOOLS_AND_MODELS.md`; each wrapped with the
latency-timing decorator from `LATENCY_AND_PERFORMANCE.md` §3.

**Dependencies**: Day 1's `Corpus`/`Document` models, `config.py`.

**Tests**: tool-level output-validation tests (§4 of `TESTING_STRATEGY.md`) for each of the five tools, plus
model-registry caching test (second call to a tool doesn't reload the model).

**Completion criteria**: each tool callable directly against a fixture corpus and returning schema-valid,
sane output; latency is recorded and non-zero for each.

## Day 3 — Agentic Multi-Step Workflows + Semantic Search

**Files**: `tools/generate_embeddings.py`, `tools/semantic_search.py`, `tools/filter_documents.py`,
`agent/nodes.py` (`understand_intent`, `plan_steps`, `execute_tool`, `route_next`, `handle_error`,
`synthesize` — real implementations replacing Day 1 stubs), `agent/graph.py` (full conditional graph wired
per `ARCHITECTURE.md` §3.3), `tests/eval_routing.py` (the agent-routing evaluation set from
`TESTING_STRATEGY.md` §3).

**Functionality**: FAISS-backed semantic search (build + query, idempotent indexing); real intent
understanding and multi-step planning/execution with conditional routing; the sentiment → filter → summarize
diagnostic chain working end-to-end; the routing eval set passing at an acceptable rate.

**Dependencies**: Day 2's tools; Day 1's LLM client and graph skeleton.

**Tests**: semantic search tests (§6), agent routing eval (§3), multi-step integration test for the
diagnostic workflow, max-iteration-guard test.

**Completion criteria**: "Why are customers unhappy?" run against a fixture reviews corpus produces a
grounded multi-step explanation with a visible tool-call trace; "find complaints about delayed delivery"
returns ranked semantic matches with sub-second query latency after indexing.

**Protected item — do not let this slip to "if time allows":** `tests/eval_routing.py` must actually be run
against the real (or mocked-deterministic) agent by end of Day 3, with a pass rate recorded. This is the
single artifact that proves the system routes rather than scripts, and it is the first thing an interviewer
who's skeptical of "agentic" claims will ask to see.

## Day 4 — Model Recommendation + Research

**⚠️ Highest-risk day in the plan.** Unlike Day 2's tools (each a single `pipeline()` call), this day has
four non-trivial, interdependent pieces: a rule engine, a real evaluation loop, a search-API integration,
and a schema that must hold all three together correctly. Budget accordingly — if something has to slip,
it should be `research_models` (research can degrade to "unavailable" gracefully), never
`evaluate_candidates` (that's the fix for the recommendation feature's core credibility gap — see
`PROJECT_SPEC.md` §11 and `MODEL_RECOMMENDATION.md` §6.5).

**Files**: `tools/model_recommendation.py` (rule engine + LLM write-up per `MODEL_RECOMMENDATION.md`),
`tools/evaluate_candidates.py` (real inference-only scoring against the user's own labeled sample — priority
#1 today), `tools/research_models.py`, `research/client.py` (`ResearchClient` interface + Tavily adapter),
`agent/workflows` wiring for recommendation intents (§7–8 of `AGENT_WORKFLOWS.md`).

**Suggested order within the day**: (1) rule-based candidate shortlist generation — cheap, deterministic,
do first; (2) `evaluate_candidates` against a labeled fixture dataset — this is today's must-land item;
(3) `research_models` + Tavily integration; (4) wire all three into `model_recommendation`'s A/B/C output.
If time runs out, stop after (2) with research degraded to "not available" rather than rushing (3)–(4) and
breaking the schema guarantees in (4).

**Functionality**: dataset-aware recommendation (rules + LLM rationale); real measured evaluation on the
user's own labeled data when available (Section A is no longer permanently empty); research retrieval with
graceful degradation on failure; the full A/B/C evidence-separated output structure implemented and
schema-enforced.

**Dependencies**: Day 3's profiling/agent infra; Day 1's `LLMClient`.

**Tests**: `evaluate_candidates` correctness test (known small labeled fixture → sane accuracy numbers,
skip-reason test for too-few-labels case), model recommendation schema/content tests (§7), research citation
propagation tests (§8), degraded-mode test (research unavailable → still produces a valid, honestly-labeled
recommendation with real Section-A numbers if labels exist).

**Completion criteria**: "Should I use BERT or DistilBERT?" against a sample **labeled** sentiment dataset
returns a correctly-labeled A/B/C response where **Section A contains real measured accuracy numbers**, B
contains source-linked evidence (or an honest "no evidence found" if the search API is unavailable), and C
is a coherent recommendation consistent with both.

## Day 5 — Frontend + Testing + Latency + Polish

**Files**: `backend/main.py` (FastAPI app: `/upload`, `/query`, `/session/{id}/history`), `frontend/app.py`
(Streamlit, per `UI_SPEC.md`), remaining test files to fill out `TESTING_STRATEGY.md` coverage,
`README.md` (per `README_PLAN.md`), latency benchmark run + results recorded, `backend/Dockerfile`,
`frontend/Dockerfile`, `docker-compose.yml`, `.dockerignore`.

**Functionality**: full FastAPI + Streamlit app wired end-to-end; workflow/status panel, latency panel,
research evidence cards, model recommendation display all implemented per `UI_SPEC.md`; benchmark runs
executed and real numbers recorded (never fabricated) into the README; **`docker compose up` starts both
services and the app is reachable exactly as it is when run locally with `uvicorn`/`streamlit run`** — this
is a final-day addition, done after the app already works locally, not a dependency for earlier days.

**Dependencies**: everything from Days 1–4.

**Tests**: FastAPI endpoint tests, full-app smoke test covering each of the 10 example use cases in
`PROJECT_SPEC.md` §6, final pass of the full test suite, a manual `docker compose up` smoke check (upload +
one query succeeds inside the containers).

**Completion criteria**: a reviewer can `uvicorn` + `streamlit run` locally, **or alternatively run
`docker compose up`**, upload each of the example files, ask each of the 10 example queries, and get a
correct, explained, latency-annotated response; README benchmarks section contains real measured numbers
with the environment they were measured on stated.

## MUST / SHOULD / NICE / DO NOT BUILD

**MUST HAVE**
- CSV/TXT/PDF ingestion + profiling
- Sentiment, classification, summarization tools
- Real LangGraph agent with conditional routing (not scripted if/else)
- One working multi-step chain (sentiment → filter → summarize)
- Semantic search with precomputed FAISS index
- **Agent routing evaluation set (§3 of `TESTING_STRATEGY.md`), actually run with a recorded pass rate** —
  promoted from SHOULD-HAVE; this is the proof the system is agentic, not scripted
- **`evaluate_candidates`: real measured accuracy/F1 on the user's own labeled data** — promoted from
  not-implemented; this is what makes Section A of the recommendation output real instead of permanently
  empty
- Dataset-aware model recommendation (rules + LLM) with correct A/B/C structure, Section A populated with
  real numbers whenever labels exist
- Per-tool and total latency measurement, surfaced in the UI
- FastAPI backend + Streamlit frontend, working together end-to-end

**SHOULD HAVE**
- Named Entity Recognition tool — demoted from MUST-HAVE to buy time for the two items above; it's the
  most commodity capability and the least connected to the project's actual differentiators
- Live research-backed recommendation with real citations
- Conversational follow-ups reusing cached prior-turn results
- LangSmith tracing wired (optional at runtime, but present in code)
- **Docker + docker-compose setup** — added after a system-design review; two-service compose (backend +
  frontend), no database/cache containers since none exist in this project's design. Built last, after the
  app already runs locally, so it never blocks core feature work.

**NICE TO HAVE**
- FAISS index persistence across process restarts (reload from disk)
- Research evidence caching within a session
- Streaming step-by-step status updates (vs. polling)
- GPU auto-utilization comparison numbers alongside CPU baseline

**DO NOT BUILD**
- Any model training / fine-tuning / LoRA / PEFT / RLHF, automatic or otherwise
- Classical ML baselines (Random Forest, SVM, regression)
- A Transformer built from scratch
- Multi-agent architectures beyond the single LangGraph agent
- Microservices decomposition / distributed infrastructure
- Enterprise authentication / multi-tenant deployment
- A React frontend (unless a concrete, demonstrated need emerges — none currently does)
