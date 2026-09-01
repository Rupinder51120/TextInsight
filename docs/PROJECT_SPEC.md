# PROJECT_SPEC.md — TextInsight

> **Note on source material:** This planning package was requested to begin with an inspection of a previously
> provided starter notebook/project. No file was found attached to this session (upload directory was empty).
> The documents below therefore incorporate the *described* flaws of that prior ML/Data-Science assistant
> (hardcoded dataset assumptions, global dataframe state, incomplete tool exposure, weak preprocessing
> architecture, limited evaluation, simple CLI interface) as explicit anti-patterns to avoid, per the
> "REUSE / REFACTOR / REMOVE" analysis in `ARCHITECTURE.md`. If you can re-upload the notebook, I can review it
> directly and tighten these documents against its actual code.

## 1. Project Overview

**TextInsight** is a conversational, tool-using system that lets a user upload
text/tabular/PDF data and ask natural-language questions about it. A LangGraph-orchestrated agent interprets
intent, decides which NLP capability (or chain of capabilities) is needed, runs deterministic Hugging Face
pipelines to produce the analysis, and — when asked about *model choice* — profiles the dataset, optionally
researches published benchmarks, and returns a reasoned, evidence-labeled recommendation.

The system is explicitly an **inference-time orchestration layer over pretrained models**, not a training
platform. Its resume-relevant novelty is in the *agentic routing, multi-step tool chaining, and
research-grounded recommendation logic* — not in model novelty.

## 2. Problem Statement

Analysts and non-ML users who have a CSV/TXT/PDF of text data and a question about it ("why are customers
unhappy?", "which model should I use?") currently have to:

- Know which NLP task (sentiment, NER, classification, summarization, semantic search) applies.
- Know which pretrained model fits their data size/latency/domain constraints.
- Manually chain multiple tools (filter negative reviews → summarize) themselves.
- Manually search papers/model cards to justify a model choice.

This project removes that manual routing burden by putting an agent in front of a curated toolbox of
deterministic NLP tools, and by making "which model and why" an answerable, evidence-separated question
rather than a guess.

## 3. Target Users

- Data analysts / PMs who need quick, explainable text analysis without writing NLP code.
- ML/NLP engineers evaluating "which pretrained model fits this dataset" before committing to fine-tuning.
- The author, as a portfolio piece demonstrating agentic system design, tool orchestration, and NLP
  engineering judgment to interviewers.

## 4. Core Value Proposition

1. **No manual pipeline selection** — natural language in, correct tool(s) selected automatically.
2. **Multi-step reasoning** — the agent can chain tools (e.g., sentiment → filter → summarize) when a single
   tool cannot answer the question.
3. **Honest model recommendation** — dataset-aware, constraint-aware, and clearly separates "measured on your
   data" from "reported by external research" from "system judgment."
4. **Measured, not claimed, performance** — every latency number shown in the UI is actually measured at
   request time; nothing is hardcoded or invented.

## 5. Complete Feature List

### NLP Capabilities
- Dataset / text profiling (row count, text length distribution, column types, language guess, label presence)
- Sentiment analysis (pretrained, 2–3 class)
- Text classification (zero-shot, user-supplied labels — no training)
- Named Entity Recognition (organizations, people, locations, misc)
- Summarization (single doc and multi-doc/batch)
- Semantic search over embedded corpus (FAISS, precomputed index)
- Deterministic filtering (e.g., "negative reviews only") between tool calls
- Multi-step workflow orchestration (agent chains 2+ tools automatically)
- Conversational follow-ups (state/memory carried across turns in a session)

### Model Guidance
- Pretrained model routing (system default model per task, latency-aware)
- Dataset-aware model/approach recommendation (rules + LLM reasoning over profiled characteristics)
- Research-backed recommendation (live web search for papers/benchmarks/model cards)
- Fine-tune-vs-pretrained advisory (recommendation only, never executed)

### Platform
- File upload: CSV, TXT, PDF
- Latency measurement and breakdown, per tool and end-to-end
- AI-generated explanations of results (LLM synthesis over deterministic tool outputs)
- Visible agent workflow/status in the UI (which tools ran, in what order)

## 6. User Journeys

**Journey A — Single-step analysis**
1. User uploads `customer_reviews.csv`.
2. User asks "Analyze the sentiment."
3. Agent profiles the text column, calls `sentiment_analysis`, returns distribution + examples + latency.

**Journey B — Multi-step diagnostic question**
1. User uploads `customer_reviews.csv`.
2. User asks "Why are customers unhappy?"
3. Agent plans: profile → sentiment → filter(negative) → topic/semantic clustering → summarize → explain.
4. Agent executes steps, showing each as it completes; returns a synthesized explanation with evidence
   (representative negative excerpts, dominant themes) and latency breakdown.

**Journey C — Model selection question**
1. User uploads a sentiment-labeled dataset.
2. User asks "Should I use BERT or DistilBERT?"
3. Agent profiles dataset (size, class balance, text length, language, latency requirement if stated).
4. Agent calls `research_models` to retrieve model-card/benchmark evidence for both candidates on comparable
   tasks.
5. Agent calls `model_recommendation` to combine profile + evidence into a labeled recommendation
   (A: nothing measured yet on this dataset because no evaluation was run; B: external benchmark citations;
   C: system recommendation and why).

**Journey D — Semantic retrieval**
1. User uploads `reviews.csv`.
2. User asks "Find complaints related to delayed delivery."
3. Agent calls `generate_embeddings` (if index doesn't exist yet) then `semantic_search` with the query,
   returns top-k matches with similarity scores and latency.

**Journey E — Conversational follow-up**
1. Continuing Journey A, user asks "Now show me only the negative ones and summarize them."
2. Agent reuses prior sentiment results from session state (no recompute), filters, summarizes.

## 7. Functional Requirements

- FR1: Accept CSV, TXT, PDF uploads up to a configured size limit; validate and parse into a normalized text
  corpus with row/document identifiers.
- FR2: Given a free-text user query and an uploaded corpus, the agent must produce an intent classification
  and a tool-execution plan without the user selecting a pipeline manually.
- FR3: The agent must support executing 1..N tools in sequence within a single user turn, where N is
  determined dynamically, not hardcoded per query type.
- FR4: Each tool call and the end-to-end request must record wall-clock latency; this must be surfaced to the
  user.
- FR5: Model recommendation responses must explicitly tag every claim as dataset-measured, external-research,
  or system-judgment.
- FR6: Research retrieval must attach source URLs/titles to any external claim used in a recommendation.
- FR7: The system must maintain conversational state within a session so follow-up queries can reference
  prior results without re-uploading or re-running unaffected tools.
- FR8: The system must never execute model training/fine-tuning regardless of user phrasing.
- FR9: Semantic search must reuse a precomputed index for a given uploaded corpus rather than re-embedding on
  every query.

## 8. Non-Functional Requirements

- **Explainability**: every result must be accompanied by an LLM-generated plain-language explanation
  grounded in the deterministic tool output (not hallucinated).
- **Determinism where possible**: NLP inference (sentiment, NER, classification, summarization, embeddings)
  runs via Hugging Face pipelines, not the LLM, to keep results reproducible and auditable.
- **Provider abstraction**: the LLM client must be swappable (Groq now, OpenAI/Anthropic later) via a single
  interface, with no orchestration code depending on a specific provider's SDK.
- **Observability**: LangSmith tracing should be attachable without being a hard dependency (system must run
  with it disabled).
- **Modularity**: each tool is an independently testable Python function/class with a typed input/output
  contract (Pydantic models).
- **No fabrication**: latency numbers, benchmark numbers, and dataset statistics must always come from actual
  measurement/computation, never from LLM invention.

## 9. Latency Goals

These are **targets to validate during development**, not guarantees, and must be reported as *measured*
numbers in the final README, not asserted in advance:

| Operation | Target (moderate input, CPU) |
|---|---|
| Dataset profiling (≤5k rows) | < 1.5s |
| Sentiment / classification / NER (batch of ~50 short texts) | 1–4s |
| Summarization (single doc, ~1–2k tokens) | 2–6s |
| Embedding generation (≤5k short texts, one-time indexing) | a few seconds, background/async where feasible |
| Semantic search query against existing index | < 300ms |
| Full multi-step workflow (3–5 chained tools + LLM synthesis) | single-digit seconds, dominated by LLM calls |

GPU availability, if any, is treated as an optional accelerator, not a requirement — the system must be
demoable on CPU-only hardware.

## 10. Scope Boundaries

**In scope:** inference over pretrained models, agentic routing/chaining, dataset-aware recommendation,
research retrieval and citation, latency instrumentation, CSV/TXT/PDF ingestion, FastAPI + Streamlit app.

**Out of scope (see also `DO NOT BUILD` in each doc):** any model training/fine-tuning/PEFT/LoRA/RLHF,
classical ML baselines (Random Forest/SVM/etc.), building a Transformer from scratch, enterprise auth,
distributed infra/microservices, multi-tenant deployment, mobile clients.

## 11. MVP Definition

**MVP (must demo end-to-end)** — rebalanced after a design review that flagged the original MVP as too
breadth-heavy and too light on the two things that actually make this more than a router around
`pipeline()` calls:
- Upload CSV/TXT/PDF → profiling.
- Single-tool queries: sentiment, classification, summarization. (NER moved to SHOULD-HAVE — see below;
  it's the most commodity, least-connected-to-the-core-story capability, and the time it costs is better
  spent protecting the two items below.)
- One working multi-step chain (sentiment → filter → summarize).
- Semantic search with precomputed FAISS index.
- **Agent routing-accuracy evaluation, run and recorded** (not just written as a test file that might not
  get run) — this is the actual proof the system is agentic and not scripted branching.
- **Dataset-aware model recommendation with real measured evaluation when labels exist**
  (`evaluate_candidates`, `MODEL_RECOMMENDATION.md` §6.5) — Section A of the recommendation output must be
  able to show an actual number, not just "no evaluation was run," on at least one of the demo datasets.
  Live web research can still degrade gracefully to "not available" — that was never the load-bearing part;
  the measured-evaluation number is.
- Latency displayed per tool and total, in the Streamlit UI.
- LangGraph agent with real conditional routing (not a hardcoded if/else pretending to be an agent).

**Full target (stretch within 5 days):**
- NER as an additional single-tool capability.
- Live research-backed recommendation with cited sources.
- Conversational follow-ups reusing prior-turn results.
- Polished Streamlit workflow visualization (step-by-step status).

**Why this reordering:** the original MVP treated all five NLP tools and the recommendation feature as
equal-priority, which meant the two hardest-to-fake, highest-credibility pieces (routing eval,
real Section-A numbers) were exactly the ones likely to get cut when time ran short on Day 4–5. Cutting one
commodity tool (NER) up front buys the time to guarantee those two survive.

See `FIVE_DAY_BUILD_PLAN.md` for the MUST/SHOULD/NICE/DO-NOT-BUILD breakdown per day.

## 12. Future Extensions (explicitly not built now)

- Optional local/offline LLM provider for full offline operation.
- User-provided labeled data → evaluation (not training) of candidate models on the user's own data, to
  upgrade "external research" claims into "measured on your dataset" claims.
- Persistent multi-session project workspace (currently session-scoped only).
- Streaming token-level responses in the UI.
- Support for additional languages beyond English-first models.
