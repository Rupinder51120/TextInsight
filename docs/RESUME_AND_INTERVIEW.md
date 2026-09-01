# RESUME_AND_INTERVIEW.md — TextInsight

## 1. Project Title (finalized)

- **"TextInsight"** — chosen final name.
- Longer/subtitle form for README or LinkedIn if useful: "TextInsight — an agentic NLP research and
  analysis assistant."

## 2. One-Line Description

"An agentic NLP platform (LangGraph + FastAPI + Streamlit) that routes natural-language questions to the
right pretrained-model pipeline — sentiment, classification, NER, summarization, semantic search — chains
tools automatically for multi-step questions, and gives evidence-separated, research-backed model
recommendations without ever training or fabricating results."

## 3. Resume Bullet Directions (write these only once each claim is true and measured)

- "Built a LangGraph agent that dynamically routes natural-language queries across 12 NLP tools (sentiment,
  classification, summarization, semantic search, model recommendation, and more), with conditional
  multi-step execution validated by an automated routing-accuracy evaluation suite (X/Y correct on a fixed
  query set — fill in once measured)."
- "Designed a dataset-aware, evidence-separated model-recommendation feature that measures candidate
  pretrained models' actual accuracy on the user's own labeled data (inference-only, zero training), and
  separates that from live-retrieved external benchmark evidence and system judgment — avoiding a common
  LLM-app failure mode (presenting unverified or external claims as if measured on the user's data)."
- "Instrumented full request-level latency tracing (per-tool, per-LLM-call, end-to-end) and optimized
  semantic search to sub-second query latency via precomputed FAISS indexing (measured: to be filled in)."
- "Shipped a containerized (Docker + Compose) FastAPI + Streamlit application end-to-end within a 5-day
  scope, using pretrained Hugging Face models exclusively at inference time (no training pipeline), with a
  provider-abstracted LLM layer (Groq, swappable)."

Do **not** finalize any bullet with a specific number until that number has actually been measured per
`LATENCY_AND_PERFORMANCE.md` §7 and `TESTING_STRATEGY.md` §3.

## 4. Technologies to Mention

Python, LangGraph, LangChain, Groq (LLM API), Hugging Face Transformers, Sentence-Transformers, PyTorch,
FAISS, FastAPI, Streamlit, Pydantic, Pandas, PyMuPDF, (LangSmith if actually used), (Tavily/search API if
actually used).

## 5. Measurable Metrics We Should Collect (mark "to be measured" until real numbers exist)

- Agent routing accuracy on the fixed evaluation query set — **to be measured**.
- Per-tool and end-to-end latency (median, cold vs. warm) for each of the 10 example use cases — **to be
  measured**.
- Semantic search: indexing time vs. query time at a stated corpus size — **to be measured**.
- Number of LLM calls per turn, by workflow type (proof of "minimize unnecessary LLM calls") — **to be
  measured**.
- Test coverage summary (unit/integration/routing-eval pass counts) — **to be measured**.

## 6. Architecture Concepts You Should Be Able to Explain

- What makes this "agentic" specifically: explicit state, nodes, conditional edges, tool-calling, and
  iterative execution — not just "we called an LLM." Be ready to describe `AgentState`, at least 3 nodes, and
  the conditional routing logic in `route_next` from memory.
- The difference between deterministic tool execution (HF pipelines) and LLM reasoning (`synthesize`,
  `model_recommendation`), and why that split exists (reproducibility, testability, hallucination
  mitigation).
- How the model-recommendation feature avoids presenting external benchmarks as if measured on the user's
  own data — the A/B/C structural contract, not just prompting — **and** how `evaluate_candidates` makes
  Section A a real, defensible number (inference-only scoring against the user's own labeled sample) rather
  than a permanent placeholder. Be ready to explain exactly why this stays "not training" (no parameter
  updates, ever).
- How semantic search achieves low query latency (precomputed embeddings, index reuse, idempotent build).
- The provider-abstraction pattern for the LLM client and why it matters (swap Groq without rewriting the
  agent).
- What was deliberately left out of scope and why (training/fine-tuning, classical ML, microservices,
  React) — being able to justify scope cuts is itself a signal of engineering judgment.

## 7. Likely Interview Questions

- "How does the agent decide which tool(s) to call? Walk me through a multi-step example."
- "How do you know the agent is actually doing multi-step reasoning and not just hardcoded branching?"
  (Answer with reference to the routing eval set and the conditional-edge implementation.)
- "How do you prevent the model recommendation from hallucinating a benchmark number?"
- "What happens if the LLM API is down mid-request?"
- "Why pretrained/zero-shot instead of fine-tuning? When would you actually recommend fine-tuning?"
- "How would you scale this to more users / bigger datasets?" (Be honest: current scope is single-session,
  in-process state; real scaling would need a persistent store, job queue for embedding/indexing, etc. —
  discuss as a known limitation/future direction, not as already solved.)
- "Why FAISS and not a managed vector DB?"
- "How did you validate the model recommendation logic isn't just the LLM making things up?" (Structural
  schema separation + rule-based candidate generation.)
- "What's the latency breakdown for a typical multi-step query, and where's the bottleneck?"

## 8. Concepts You Must Be Able to Explain (beyond the project itself)

- Transformer attention basics, contextual embeddings, difference between encoder-only (BERT-family, used
  here for classification/NER/sentiment) and the summarization models' encoder-decoder architecture.
- Zero-shot classification via NLI models — how framing "does this text entail this label" enables
  classification without task-specific training.
- Sentence embeddings vs. token embeddings, and why pooled sentence embeddings are what FAISS indexes.
- Model size vs. latency tradeoffs (why distilled models were chosen as defaults).
- What LangGraph adds over a plain LangChain agent/tool loop (explicit typed state, graph structure,
  conditional edges, easier testing of individual nodes).

## 9. Claims We Should NOT Make on the Resume

- Do not claim the system was "benchmarked" against other models unless an actual evaluation was run and
  numbers recorded.
- Do not claim "state-of-the-art" results — none were established.
- Do not claim fine-tuning capability the system doesn't have.
- Do not claim production-scale/enterprise readiness (no auth, no multi-tenancy, no distributed infra, by
  design).
- Do not claim specific latency numbers not actually measured in this build.
- Do not claim the routing/recommendation logic is "fully autonomous" without qualification — it operates
  within a bounded tool catalog and iteration cap, by design, and that boundedness is a feature, not a
  weakness, but should be described accurately.
